# 엣지 ↔ 서버 ↔ 웹 API

라즈베리파이(엣지)와 서버가 주고받는 형식이에요. `server/app.py`(개발 서버)가 이 형식을 따르고,
나중에 PostgreSQL·FCM을 붙인 정식 서버도 같은 형식을 지키면 엣지 코드를 바꿀 필요가 없어요.
보안(인증·TLS)은 이번 범위에서 빠져 있어요. 개발 서버를 인터넷에 공개하지 마세요.

시각은 모두 유닉스 시각(초, 소수점 포함)이에요.

## 시간 규칙

| 항목 | 값 | 담당 |
|---|---|---|
| 위험 기준 시간 | 엎드림·얼굴 가림이 5초 이어지면 위험 (서버 설정으로 변경). 1초 이하로 잠깐 안 잡히는 건 이어진 것으로 봄 | 엣지 `sleepsafe/risk.py` |
| 위험 종료 | 정상 판정이 3초 이어지면 종료 (보호자 확인으로는 종료 안 됨) | 엣지 |
| 1차 알림 | 위험 시작을 받으면 바로 | 서버 |
| 2차 알림 | 1차 후 15초 동안 확인이 없고 위험이 계속되면 모든 보호자에게. 그 전에 위험이 끝나면 보내지 않음 | 서버 |
| 생존 신호 | 엣지가 2초마다 전송 | 엣지 `sleepsafe/notify.py` |
| 연결 끊김 | 마지막 생존 신호 후 8초 | 서버 |

## 엣지 → 서버

### `POST /devices/{device_id}/heartbeat`
2초마다 현재 상태를 보내요. 응답의 `settings`로 서버의 최신 설정을 받아요.
서버에 연결되면 **서버 설정이 엣지 설정(configs/edge.yaml)보다 우선**해요.

```json
{
  "ts": 1760342400.12,
  "status": "normal",            // normal 정상 / caution 주의 / risk 위험 / unavailable 분석 불가
  "reason": "",                  // 주의·위험: prone / face_covered, 분석 불가: no_baby / dark
                                 // (covered_without_body일 때 몸 없이 가린 얼굴만 보이면 body_not_found)
  "posture": "supine",           // supine / side / prone / null
  "airway": "airway_visible",    // airway_visible / airway_covered / null
  "elapsed": {"prone": 0.0, "face_covered": 0.0},   // 조건별로 이어진 시간(초)
  "active_event_id": null,
  "fps": 2.9, "infer_ms": 210.5,
  "model": "A_original_416.onnx",
  "settings_version": 3,         // 엣지에 실제로 적용된 설정 버전
  "dropped_events": 0            // 연결이 끊긴 동안 큐가 넘쳐 버린 이벤트 수
}
```
응답:
```json
{"ok": true, "settings": {"version": 3, "risk_threshold_s": 5.0, "escalation_s": 15.0}}
```

### `POST /devices/{device_id}/events`
같은 `event_id`로 다시 보내도 알림은 한 번만 나가요 (네트워크 재전송 대비).

위험 시작:
```json
{
  "event_id": "3f2c9a...",
  "kind": "risk_started",
  "risk_type": "prone",          // prone 엎드림 / face_covered 얼굴 가림
  "started_at": 1760342395.10,   // 조건이 처음 관찰된 시각
  "confirmed_at": 1760342400.12, // 위험으로 확정된 시각
  "posture": "prone", "airway": "airway_covered", "confidence": 0.87,
  "model": "A_original_416.onnx", "settings_version": 3,
  "snapshot_jpeg_b64": "/9j/4AAQ..."   // 선택. 위험 순간 사진 (원본 영상은 보내지 않음)
}
```
위험 종료:
```json
{"event_id": "3f2c9a...", "kind": "risk_ended", "risk_type": "prone", "ended_at": 1760342431.40}
```

## 웹(보호자) → 서버

| 요청 | 설명 |
|---|---|
| `GET /devices` | 모든 카메라 상태 (`online`, `age_s`, `settings_applied` 포함) |
| `GET /devices/{id}/status` | 카메라 하나의 상태 |
| `GET /devices/{id}/risk-settings` | 설정 보기 |
| `PUT /devices/{id}/risk-settings` | 설정 바꾸기 `{"risk_threshold_s": 5, "escalation_s": 15}` → 버전이 1 올라가고, 엣지가 다음 생존 신호 때 받아서 적용 |
| `GET /events` | 위험 이벤트 목록 (최신순) |
| `GET /events/{event_id}/snapshot` | 스냅샷 JPEG |
| `POST /events/{event_id}/ack` | 보호자 확인 `{"guardian": "엄마"}`. 2차 알림만 막고 위험 상태는 그대로 |
| `GET /alerts` | 알림 기록 (최신순) |
| `GET /stream` | 실시간 알림 (Server-Sent Events) |

`/stream`으로 오는 메시지:
```json
{"type": "alert", "alert": {"level": 2, "kind": "risk_escalated", "message": "...", "targets": "all", ...}}
{"type": "device", "device": {...}}
{"type": "event", "event": {...}}
{"type": "settings", "device_id": "crib-01", "settings": {...}}
```

알림 `kind`: `risk`(1차, level 1), `risk_escalated`(2차, level 2), `risk_ended`(level 0),
`connection_lost`(level 1), `connection_restored`(level 0)

## 나중에 바꿀 곳
- 저장: `server/app.py`의 `State` → PostgreSQL
- 발송: `publish()`에 FCM(모바일)·Web Push(브라우저를 닫아도 알림) 추가
- 보안: 기기 인증 토큰, 보호자 로그인, HTTPS
