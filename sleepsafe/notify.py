"""엣지 → 서버 전송 (웹 알림용). 요청 형식은 docs/api.md에 있어요.

HttpNotifier는 별도 스레드에서 보내서, 네트워크가 느리거나 끊겨도 추론이 멈추지 않아요.
보내지 못한 이벤트는 최대 max_queue개까지 들고 있다가 다시 연결되면 보내요.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

log = logging.getLogger("sleepsafe.notify")


class ConsoleNotifier:
    """서버 없이 콘솔에만 출력."""

    settings: Optional[Dict[str, Any]] = None
    dropped = 0

    def send_event(self, payload: Dict[str, Any]) -> None:
        shown = {k: v for k, v in payload.items() if k != "snapshot_jpeg_b64"}
        print("[이벤트]", json.dumps(shown, ensure_ascii=False))

    def update_status(self, payload: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class HttpNotifier:
    def __init__(self, base_url: str, device_id: str, heartbeat_s: float = 2.0,
                 timeout: float = 3.0, max_queue: int = 100):
        import requests

        self._http = requests.Session()
        self._errors = (requests.RequestException, ValueError)
        self.base = base_url.rstrip("/")
        self.device_id = device_id
        self.heartbeat_s = heartbeat_s
        self.timeout = timeout
        self.max_queue = max_queue
        self.settings: Optional[Dict[str, Any]] = None  # 서버가 내려준 최신 설정
        self.dropped = 0                                # 큐가 넘쳐 버린 이벤트 수
        self._events: deque = deque()
        self._status: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="notifier", daemon=True)
        self._thread.start()

    def send_event(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            if len(self._events) >= self.max_queue:
                self._events.popleft()
                self.dropped += 1
            self._events.append(payload)
        self._wake.set()

    def update_status(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._status = dict(payload)

    def close(self, flush_timeout: float = 3.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=flush_timeout)

    # 아래는 전송 스레드에서 실행
    def _url(self, path: str) -> str:
        return f"{self.base}/devices/{self.device_id}/{path}"

    def _run(self) -> None:
        next_heartbeat = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_heartbeat:
                self._heartbeat()
                next_heartbeat = now + self.heartbeat_s
            self._flush()
            self._wake.wait(timeout=0.2)
            self._wake.clear()
        self._flush()      # 끝나기 전에 남은 이벤트와
        self._heartbeat()  # 마지막 상태를 보내요

    def _heartbeat(self) -> None:
        with self._lock:
            payload = dict(self._status, dropped_events=self.dropped)
        try:
            r = self._http.post(self._url("heartbeat"), json=payload, timeout=self.timeout)
            r.raise_for_status()
            self.settings = r.json().get("settings")
        except self._errors as e:
            log.warning("생존 신호 전송 실패: %s", e)

    def _flush(self) -> None:
        while True:
            with self._lock:
                if not self._events:
                    return
                payload = self._events[0]
            try:
                r = self._http.post(self._url("events"), json=payload, timeout=self.timeout)
                r.raise_for_status()
            except self._errors as e:
                log.warning("이벤트 전송 실패 (나중에 다시 보냄): %s", e)
                return
            with self._lock:
                if self._events and self._events[0] is payload:
                    self._events.popleft()
