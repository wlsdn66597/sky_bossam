"""웹 알림 개발 서버 (보안 없음·메모리 저장 → 개발·시연 전용, 인터넷에 공개하지 마세요).

실행:     uvicorn server.app:app --host 0.0.0.0 --port 8000
대시보드:  http://<서버IP>:8000
API 형식:  docs/api.md

알림 규칙 (아키텍처 명세서 3.3절, 8장)
- 위험 시작 → 1차 알림
- 1차 알림 후 escalation_s(기본 15초) 동안 확인이 없고 위험이 계속되면 → 2차 알림 (모든 보호자)
- 2차 알림 전에 위험이 끝나면 2차 알림은 보내지 않음
- 생존 신호가 8초 동안 없으면 연결 끊김 알림
나중에 PostgreSQL 저장, FCM·Web Push 발송으로 바꿀 자리는 State와 publish()예요.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

OFFLINE_AFTER_S = 8.0
MAX_ALERTS = 200
MAX_SNAPSHOT_BYTES = 2_000_000
STATIC = Path(__file__).parent / "static"
RISK_NAMES = {"prone": "엎드림", "face_covered": "얼굴 가림"}


class RiskSettings(BaseModel):
    risk_threshold_s: float = Field(5.0, ge=1, le=60, description="위험 기준 시간(초)")
    escalation_s: float = Field(15.0, ge=0.1, le=600, description="1차 알림 후 2차 알림까지(초)")


class EventIn(BaseModel):
    event_id: str
    kind: str                           # risk_started / risk_ended
    risk_type: Optional[str] = None     # prone / face_covered
    started_at: Optional[float] = None
    confirmed_at: Optional[float] = None
    ended_at: Optional[float] = None
    posture: Optional[str] = None
    airway: Optional[str] = None
    confidence: Optional[float] = None
    settings_version: Optional[int] = None
    model: Optional[str] = None
    snapshot_jpeg_b64: Optional[str] = None


class AckIn(BaseModel):
    guardian: str = "보호자"


class State:
    def __init__(self) -> None:
        self.devices: Dict[str, dict] = {}
        self.settings: Dict[str, dict] = {}
        self.events: Dict[str, dict] = {}
        self.snapshots: Dict[str, bytes] = {}
        self.alerts: List[dict] = []
        self.subscribers: Set[asyncio.Queue] = set()
        self.tasks: Set[asyncio.Task] = set()


state = State()


def get_settings(device_id: str) -> dict:
    return state.settings.setdefault(device_id, {"version": 1, **RiskSettings().model_dump()})


def device_view(d: dict) -> dict:
    s = get_settings(d["device_id"])
    return {**d, "age_s": round(time.time() - d["last_seen"], 1),
            "settings_applied": d.get("settings_version") == s["version"], "settings": s}


def publish(msg: dict) -> None:
    """대시보드(SSE)로 보내요. FCM·Web Push를 붙일 때도 여기서 같이 보내면 돼요."""
    for q in list(state.subscribers):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


def add_alert(kind: str, level: int, device_id: str, message: str, event_id: Optional[str] = None,
              risk_type: Optional[str] = None, targets: str = "primary") -> dict:
    """level 2: 2차(모든 보호자), 1: 1차·연결 끊김, 0: 안내"""
    alert = {"alert_id": uuid.uuid4().hex, "ts": time.time(), "kind": kind, "level": level,
             "device_id": device_id, "event_id": event_id, "risk_type": risk_type,
             "message": message, "targets": targets}
    state.alerts.append(alert)
    del state.alerts[:-MAX_ALERTS]
    publish({"type": "alert", "alert": alert})
    return alert


def event_view(ev: dict) -> dict:
    return dict(ev, has_snapshot=ev["event_id"] in state.snapshots)


async def escalate_later(event_id: str, delay: float) -> None:
    await asyncio.sleep(delay)
    ev = state.events.get(event_id)
    if ev and ev["status"] == "active" and not ev["acked"]:
        ev["escalated"] = True
        name = RISK_NAMES.get(ev["risk_type"], ev["risk_type"])
        add_alert("risk_escalated", 2, ev["device_id"],
                  f"{name} 위험이 {delay:g}초 동안 확인되지 않았어요 (2차 알림: 모든 보호자)",
                  event_id, ev["risk_type"], targets="all")
        publish({"type": "event", "event": event_view(ev)})


async def watchdog() -> None:
    while True:
        now = time.time()
        for device_id, d in list(state.devices.items()):
            if d["online"] and now - d["last_seen"] > OFFLINE_AFTER_S:
                d["online"] = False
                add_alert("connection_lost", 1, device_id, "카메라 연결이 끊겼어요 (생존 신호 없음)")
                publish({"type": "device", "device": device_view(d)})
        await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(watchdog())
    yield
    task.cancel()


app = FastAPI(title="질식 위험 알림 개발 서버", lifespan=lifespan)


@app.post("/devices/{device_id}/heartbeat")
async def heartbeat(device_id: str, payload: Dict[str, Any] = Body(default_factory=dict)):
    d = state.devices.get(device_id)
    was_offline = d is not None and not d["online"]
    if d is None:
        d = state.devices[device_id] = {"status": "unavailable", "reason": "starting"}
    d.update(payload)
    d.update(device_id=device_id, last_seen=time.time(), online=True)
    if was_offline:
        add_alert("connection_restored", 0, device_id, "카메라 연결이 복구됐어요")
    publish({"type": "device", "device": device_view(d)})
    return {"ok": True, "settings": get_settings(device_id)}


@app.get("/devices")
def list_devices():
    return [device_view(d) for d in state.devices.values()]


@app.get("/devices/{device_id}/status")
def device_status(device_id: str):
    d = state.devices.get(device_id)
    if d is None:
        raise HTTPException(404, "등록되지 않은 기기예요")
    return device_view(d)


@app.get("/devices/{device_id}/risk-settings")
def read_settings(device_id: str):
    return get_settings(device_id)


@app.put("/devices/{device_id}/risk-settings")
def write_settings(device_id: str, body: RiskSettings):
    s = get_settings(device_id)
    s.update(body.model_dump())
    s["version"] += 1
    publish({"type": "settings", "device_id": device_id, "settings": s})
    return s


@app.post("/devices/{device_id}/events")
async def post_event(device_id: str, ev: EventIn):
    existing = state.events.get(ev.event_id)
    name = RISK_NAMES.get(ev.risk_type or "", ev.risk_type or "위험")

    if ev.kind == "risk_started":
        if existing is not None:
            return {"ok": True, "duplicate": True}  # 재전송 → 같은 알림을 두 번 보내지 않음
        if ev.snapshot_jpeg_b64:
            try:
                data = base64.b64decode(ev.snapshot_jpeg_b64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(422, "snapshot_jpeg_b64가 올바른 base64가 아니에요")
            if len(data) <= MAX_SNAPSHOT_BYTES:
                state.snapshots[ev.event_id] = data
        record = ev.model_dump(exclude={"snapshot_jpeg_b64", "kind"})
        record.update(device_id=device_id, status="active", acked=False, acked_at=None, acked_by=None,
                      escalated=False, received_at=time.time())
        state.events[ev.event_id] = record
        add_alert("risk", 1, device_id, f"{name} 위험 감지 (1차 알림)", ev.event_id, ev.risk_type)
        task = asyncio.create_task(escalate_later(ev.event_id, get_settings(device_id)["escalation_s"]))
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)
        publish({"type": "event", "event": event_view(record)})
        return {"ok": True}

    if ev.kind == "risk_ended":
        if existing is None:  # 시작 이벤트를 못 받은 경우에도 기록은 남김
            existing = state.events[ev.event_id] = ev.model_dump(exclude={"snapshot_jpeg_b64", "kind"})
            existing.update(device_id=device_id, status="active", acked=False, acked_at=None,
                            acked_by=None, escalated=False, received_at=time.time())
        elif existing["status"] == "ended":
            return {"ok": True, "duplicate": True}
        existing.update(status="ended", ended_at=ev.ended_at)
        note = "" if existing["acked"] else " (보호자 미확인 상태로 해소)"
        add_alert("risk_ended", 0, device_id, f"{name} 위험이 해소됐어요{note}", ev.event_id, ev.risk_type)
        publish({"type": "event", "event": event_view(existing)})
        return {"ok": True}

    raise HTTPException(422, "kind는 risk_started 또는 risk_ended여야 해요")


@app.get("/events")
def list_events(limit: int = 50):
    evs = sorted(state.events.values(), key=lambda e: e["received_at"], reverse=True)
    return [event_view(e) for e in evs[:limit]]


@app.get("/events/{event_id}/snapshot")
def event_snapshot(event_id: str):
    data = state.snapshots.get(event_id)
    if data is None:
        raise HTTPException(404, "스냅샷이 없어요")
    return Response(content=data, media_type="image/jpeg")


@app.post("/events/{event_id}/ack")
def ack_event(event_id: str, body: Optional[AckIn] = None):
    """보호자 확인 = 상황을 인지했다는 뜻. 위험 상태를 끝내지는 않아요 (2차 알림만 막음)."""
    ev = state.events.get(event_id)
    if ev is None:
        raise HTTPException(404, "없는 이벤트예요")
    if not ev["acked"]:
        ev.update(acked=True, acked_at=time.time(), acked_by=(body.guardian if body else "보호자"))
        publish({"type": "event", "event": event_view(ev)})
    return event_view(ev)


@app.get("/alerts")
def list_alerts(limit: int = 50):
    return list(reversed(state.alerts[-limit:]))


@app.get("/stream")
async def stream():
    """Server-Sent Events: 대시보드가 알림을 실시간으로 받아요."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    state.subscribers.add(q)

    async def gen():
        try:
            yield "retry: 3000\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            state.subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
