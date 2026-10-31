# sky_bossam · 영유아 질식 위험(엎드림·얼굴 가림) 탐지

침대 위 카메라 영상에서 **엎드림**과 **얼굴(코·입) 가림**을 찾아, 5초 이상 이어지면 보호자에게 알리는 시스템이에요.
이 레포에는 다음이 들어 있어요.

1. **모델 학습·평가** (RTX 4090 PC): YOLOv8n 탐지 모델, 데이터셋 정리, 실험 자동 실행
2. **실시간 실행** (라즈베리파이 4 / 노트북): ONNX Runtime 추론, 5초 위험 판정, 서버 전송
3. **웹 알림 개발 서버**: 1차·2차 알림, 연결 끊김 감지, 보호자 대시보드

> 의료기기가 아니에요. 영상에서 보이는 위험 징후를 알려 보호자의 확인을 돕는 시제품이에요.
> 보안(인증·암호화)은 아직 범위 밖이라 개발 서버를 인터넷에 공개하면 안 돼요.

```
카메라 ─▶ [라즈베리파이 4]  탐지(YOLO, ONNX) → 판정(sleepsafe/decision.py) → 5초 타이머(sleepsafe/risk.py)
                │  생존 신호(2초) · 위험 시작/종료 · 스냅샷
                ▼
          [서버]  1차 알림 → 15초 미확인 시 2차 알림 · 8초 무응답 시 연결 끊김
                │  Server-Sent Events
                ▼
          [웹 대시보드]  상태 · 스냅샷 · 확인 버튼 · 기준 시간 설정
```

PC에서 정확도를 잴 때와 파이에서 실제로 돌릴 때 **같은 판정 코드**(`sleepsafe/decision.py`)를 써요.
그래서 PC에서 나온 정확도가 파이의 동작과 어긋나지 않아요.

---

## 데이터셋

**[Baby-Position](https://universe.roboflow.com/jennifer-verona-uzduw/baby-position-esrva)** (Roboflow Universe, v1, 2,745장, CC BY 4.0)

- 위에서 내려다본 아기·인형 사진에 **몸 박스(자세)**와 **얼굴 박스(코·입 상태)**가 함께 라벨링돼 있어요.
- v1은 학습 2,198 / 검증 274 / 테스트 273장이에요. 모든 사진이 **512×512로 늘려 맞춰져** 있어서 입력 크기는 512가 최대예요. 압축 파일은 대략 100~200MB예요.
- 여러 공개 데이터셋을 모은 것이에요. 아래 데이터셋은 이미 들어 있어서, 따로 추가하면 같은 사진이 중복돼요. 외부 테스트로도 쓸 수 없어요.
  - [SIDS (kjy)](https://universe.roboflow.com/kjy-5agtz/sids-vrari): `IMG_` 출처 1,462장이 이 데이터셋이에요. 파일명과 픽셀이 같은 걸 확인했어요.
  - [Baby-Detection](https://universe.roboflow.com/home-zg10t/baby-detection-wabjm): 실제 아기 사진(`Prone_`·`Sideways_`·`Supine_`)
  - [skt/baby_posture](https://universe.roboflow.com/skt/baby_posture): 일부(약 30장)만 들어 있어요.
- 사진은 레포에 넣지 않아요. 아래 방법으로 각자 받아요.

원본 클래스는 `scripts/prepare_dataset.py`가 아래처럼 정리해요 (`configs/dataset.yaml`).

| 원본 | 우리 클래스 | 박스 위치 | 뜻 |
|---|---|---|---|
| Baby-Supine | `supine` | 몸 전체 | 바로 누움 |
| Baby-Sideways | `side` | 몸 전체 | 옆으로 누움 |
| Baby-Prone | `prone` | 몸 전체 | 엎드림 → **위험 조건** |
| Airway-Visible | `airway_visible` | 얼굴 | 코·입 보임 |
| Airway-Covered | `airway_covered` | 얼굴 | 코·입 가림 → **위험 조건** |
| Active-Parent, Toy-Beside | (버림) | | |

사진 출처는 파일명 패턴으로 구분해요. 출처별 분할(실험 B)에 써요.

| 출처 이름 | 파일명 | 내용 |
|---|---|---|
| `skt_doll` | `1190_png...` | skt/baby_posture 인형 사진 (흑백 야간 포함) |
| `real_baby` | `Prone_12...`, `Sideways_...`, `Supine_...` | 실제 아기 사진 |
| `doll_blanket` | `IMG_8761...` | SIDS(kjy) 인형 연속 촬영. 회색 천으로 덮은 사진 포함. 엎드리지 않은 얼굴 가림 사진은 거의 여기에만 있어요 |
| `bassinet` | `06695D11-...` | 바구니 침대를 위에서 찍은 사진 |
| `other` | 그 외 | |

---

## 1. 학습 PC (RTX 4090) 준비

```bash
git clone https://github.com/wlsdn66597/sky_bossam.git
cd sky_bossam
python -m venv .venv
```
가상환경 켜기: Windows는 `.venv\Scripts\activate`, Linux는 `source .venv/bin/activate`

1. **PyTorch를 먼저** 설치해요. 순서가 바뀌면 GPU가 안 잡히는 CPU 버전이 깔릴 수 있어요.
   - Linux: `pip install torch torchvision` (Linux용 기본 패키지에 CUDA가 들어 있어요)
   - Windows: [pytorch.org](https://pytorch.org/get-started/locally/)에서 CUDA 버전용 설치 명령을 복사해요.
2. 나머지를 설치하고 GPU가 잡히는지 확인해요. `True NVIDIA GeForce RTX 4090`이 나오면 돼요.
```bash
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. 데이터 받기

[Roboflow](https://app.roboflow.com)에 가입해요. 다음 두 방법 중 하나를 쓰면 돼요.

**방법 A. curl (브라우저 없는 원격 서버에 편해요)**

데이터셋 페이지 → Download Dataset → 형식 **YOLOv8** → Show download code → **Terminal**에 나오는 링크를 써요.
Roboflow가 보여 주는 명령은 현재 폴더에 바로 압축을 풀어요. 그래서 아래처럼 받을 위치를 정해 주세요.
링크에 키가 들어 있으니 공유하거나 깃에 올리지 마세요.
```bash
curl -L "Roboflow가_보여준_링크" --create-dirs -o datasets/raw/roboflow.zip
python -m zipfile -e datasets/raw/roboflow.zip datasets/raw/baby-position-v1
```
Windows PowerShell에서는 `curl` 대신 `curl.exe`라고 써야 해요.
압축을 풀고 나면 `datasets/raw/baby-position-v1/data.yaml`이 있어야 해요.

**방법 B. API 키**

Settings → API Keys에서 Private API Key를 복사해 환경변수로 넣어요.
```bash
ROBOFLOW_API_KEY=여기에_키 python scripts/download_dataset.py
```
Windows PowerShell: `$env:ROBOFLOW_API_KEY="여기에_키"; python scripts/download_dataset.py`

## 3. 데이터 정리

```bash
python scripts/prepare_dataset.py
```
- 출처별·분할별 박스 수가 표로 나와요. 테스트로 뺀 출처에 없는 클래스가 있으면 경고가 떠요.
- 다른 출처도 테스트로 빼 보려면 `--holdout doll_blanket real_baby`처럼 적어요.
- 그다음 `configs/experiments.yaml`에 그 실험 줄을 추가하면 돼요.

## 4. 실험 실행

먼저 3에폭만 짧게 돌려서 끝까지 도는지 확인해요.
```bash
python scripts/run_experiments.py --only A_original_416 --set epochs=3
```
그다음 전체 실험을 돌려요. 4090 기준으로 실험 하나에 대략 30분~1시간 걸려요.
```bash
python scripts/run_experiments.py
```
원격(SSH)으로 돌린다면, 접속이 끊겨도 학습이 계속되게 백그라운드로 돌리고 로그를 보세요 (Linux).
```bash
nohup python scripts/run_experiments.py > train.log 2>&1 &
```
```bash
tail -f train.log
```

| 실험 | 내용 | 알 수 있는 것 |
|---|---|---|
| `A_original_512/416/320` | 원래 분할, 입력 크기별 | 기본 성능, 파이 4용으로 입력을 얼마나 줄여도 되는지 |
| `B_doll_blanket_416/320` | SIDS(kjy) 인형 사진(천으로 덮은 사진 포함)을 통째로 테스트로 | **처음 보는 인형**에서의 성능 (우리 인형 테스트에 가까운 추정) |
| `A_original_416_aug` | 좌우 반전·밝기 변형 켬 | 증강이 도움이 되는지 |
| `A_original_320_aug` | 320 + 증강 | 파이 4에 올릴 후보 |

**A 점수를 읽을 때 주의할 점:** 원래 분할에는 연속 촬영 사진이 학습과 테스트에 섞여 있어요.
그래서 A 점수는 실제보다 높게 나와요. A와 B의 차이가 바로 그 부풀려진 정도예요.

학습 설정은 `configs/train.yaml`에 있어요. Pirotello 외(SBCUP 2026)의 설정을 따랐어요.
- YOLOv8n, 200에폭, Adam, 학습률 0.001 고정, 좌우 반전·밝기 변형 끔
- 배치는 16이에요. 4090이면 `--set batch=64`로 늘려 더 빨리 돌릴 수도 있어요. 단, 배치를 바꾸면 논문과 조건이 달라져요.

### 결과 보기
`results/summary.csv`(엑셀로 열림), 실험별 상세는 `results/exp_<이름>.json`

| 열 | 뜻 |
|---|---|
| `mAP50` | 박스 탐지 종합 점수 |
| `prone_P/R`, `covered_P/R` | 엎드림·코입가림 박스의 정밀도/재현율 |
| `posture_acc` | 사진마다 자세(바로/옆/엎드림)를 맞힌 비율 |
| `risk_acc` | 사진마다 위험 여부(엎드림 또는 코·입 가림)를 맞힌 비율 |
| `risk_recall` | 위험한 사진 중 위험으로 잡은 비율 → **가장 중요** (놓치면 안 됨) |
| `risk_precision` | 위험이라고 한 것 중 실제로 위험한 비율 (낮으면 잘못된 알림이 많음) |
| `infer_ms` | 학습 PC에서의 장당 추론 시간 (파이 4는 훨씬 느려요) |

박스 혼동행렬 같은 그래프는 `runs/val/<이름>/`에 저장돼요.
파이와 똑같은 ONNX 추론 코드로 다시 평가하려면 이렇게 실행해요.
```bash
python scripts/evaluate.py --weights models/A_original_320_aug.onnx --backend onnx \
    --data datasets/sleepsafe/data_original.yaml --imgsz 320
```

### 판정 기준 정하기 (재학습 없이)
사진 단위 판정은 `configs/edge.yaml`의 기준값을 써요.
- 몸 박스 신뢰도 `conf_body`, 코·입 박스 신뢰도 `conf_airway`
- 몸 박스 없이 코·입 가림만 잡혀도 인정할지 `covered_without_body`

기준이 너무 엄격하면 정밀도는 높고 재현율은 낮게 나와요.
아래 명령은 예측을 한 번만 돌리고, 84개 조합을 다시 판정해서 비교해요.
```bash
python scripts/sweep_thresholds.py --exp A_original_416 A_original_320
```
지금 기본값(몸 0.2, 얼굴 0.15, 몸 없이 가림 인정)은 `A_original_416_aug` 비교 결과로 골랐어요.
- 처음 값(0.5 / 0.4): 위험 재현율 66.7%
- 지금 값: 위험 재현율 93.6%, 정밀도 97.1%

5초 규칙이 몇 프레임짜리 오탐은 걸러 주니, 기준은 놓치지 않는 쪽(재현율)으로 기울이는 게 맞아요.
- 지금 설정, 추천 조합(위험 재현율 90% 이상 중 정밀도 최고), 재현율 상위 10개가 나와요.
- 전체 조합은 `results/sweep_<실험>_<백엔드>.csv`에 저장돼요.

추천 값을 `configs/edge.yaml`에 넣고, 재학습 없이 평가만 다시 돌리면 `summary.csv`가 새 기준으로 바뀌어요.
```bash
python scripts/run_experiments.py --skip-train --no-export
```
테스트 세트로 고른 기준이라 테스트 점수가 실제보다 약간 좋게 나와요. 직접 찍은 영상으로 다시 확인하세요.

### 다른 데이터셋 전체로 평가하기
직접 찍은 테스트 세트처럼 다른 데이터셋을 통째로 평가할 때는 `--split all`을 써요.
- 학습·검증·테스트로 나뉜 사진을 모두 합쳐서 사진 단위 판정만 평가해요.
- 데이터셋 설정 파일은 `configs/dataset.yaml`을 복사해서 `roboflow`, `raw_dir`, `out_dir`, `class_map`만 바꾸면 돼요.
```bash
python scripts/evaluate.py --weights runs/train/A_original_320_aug/weights/best.pt \
    --data datasets/<데이터셋>/data_original.yaml --imgsz 320 --split all --name <결과이름>
```
기준값 비교도 같은 옵션(`--weights --data --imgsz --split all`)을 `sweep_thresholds.py`에 넘기면 돼요.

---

## 5. 노트북에서 실시간 실행 (웹캠·영상)

```bash
pip install -r requirements-edge.txt
python edge/run_edge.py --source 0 --model models/A_original_416.onnx --imgsz 416 --show
python edge/run_edge.py --source 녹화.mp4 --model models/A_original_416.onnx --imgsz 416
```
- 영상 파일은 영상 안의 시간을 기준으로 5초를 재요. 그래서 빨리 돌려도 판정 결과는 같아요.
- 판정 기준(신뢰도, 밝기, 5초·3초)은 `configs/edge.yaml`에 있어요.
- `miss_tolerance_s`(기본 1초): 엎드림·얼굴 가림이 이보다 짧게 안 잡히는 건 탐지가 깜빡인 것으로 보고 5초 타이머를 유지해요. 0이면 한 프레임만 놓쳐도 타이머가 처음부터 다시 시작해요.

## 6. 웹 알림 서버

```bash
pip install -r requirements-server.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
- 브라우저에서 `http://localhost:8000`을 열어요.
- 엣지를 `--server http://<서버 PC의 IP>:8000`으로 실행하면 대시보드에 상태와 알림이 떠요.
- "브라우저 알림 켜기"를 누르면 탭이 뒤에 있어도 알림이 와요.
- 다른 기기에서 접속이 안 되면 서버 PC의 방화벽에서 8000번 포트를 열어 주세요.
- 요청 형식과 알림 규칙: [docs/api.md](docs/api.md)

## 7. 라즈베리파이 4로 옮기기

64비트 Raspberry Pi OS를 써야 해요 (onnxruntime이 64비트용으로 배포돼요).
numpy 버전 충돌을 피하려고 카메라·OpenCV는 apt로, onnxruntime만 pip로 설치해요.

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-yaml python3-requests
git clone https://github.com/wlsdn66597/sky_bossam.git && cd sky_bossam
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install onnxruntime
```
학습 PC에서 만든 모델을 파이의 `models/`로 옮겨요. 파이에서 학습 PC로 접속할 수 있으면 `scp`로 바로 가져와요.
```bash
mkdir -p models
scp 사용자@학습PC주소:~/sky_bossam/models/A_original_320_aug.onnx models/
```
**1) 모델 속도만 재기 (카메라 없이)** — 여러 모델을 한 번에 비교할 수 있어요.
```bash
python edge/benchmark.py --model models/A_original_320_aug.onnx models/A_original_416_aug.onnx
```
**2) 카메라로 실행** — 10초마다 평균 추론 시간과 처리 속도(fps)가 출력돼요. `--max-seconds 60`으로 1분만 돌려 볼 수 있어요.
```bash
python edge/run_edge.py --source picamera2 --model models/A_original_320_aug.onnx --imgsz 320 \
    --server http://<서버 IP>:8000
```
- **입력 크기:** 파이 4는 느려서 입력 320이나 416을 추천해요. Ultralytics 측정으로 YOLOv8n 640 입력은 ONNX 기준 장당 약 560ms예요. `configs/edge.yaml`의 `model`, `imgsz`도 맞춰 주세요.
- **속도 기준:** 추론 평균이 300ms 이하면 초당 3번 판정(`infer_fps: 3`)이 가능해요. 더 느리면 `infer_fps`를 2로 낮춰도 5초 판정에는 충분해요.
- **카메라 설치:** 침대 위에서 수직으로 내려다보게 설치해요. 학습 데이터가 그 각도예요.
- **처리 속도:** 카메라 수집과 추론은 따로 돌아가요. 추론은 초당 3번(`infer_fps`)이면 5초 판정에 충분해요.

---

## 폴더 구조

```
sleepsafe/         공통 코드 (파이에서도 그대로 사용, PyTorch 불필요)
  decision.py        탐지 결과 → 자세·코입 상태·위험 유형 (평가와 실행이 같이 씀)
  risk.py            5초 위험 판정, 3초 정상 종료, 분석 불가 처리
  detector_onnx.py   ONNX Runtime 추론 (Ultralytics와 같은 전처리·후처리)
  notify.py          서버 전송 (별도 스레드, 끊기면 최대 100건 보관)
  sources.py         웹캠·영상 파일·파이 카메라 입력
  imaging.py         한글 경로 이미지 읽기, 밝기, 스냅샷, 화면 표시
scripts/           학습 PC용: 받기 → 정리 → 학습 → 평가 → 내보내기 → 실험 자동 실행, 판정 기준 비교
edge/run_edge.py   실시간 실행 (10초마다 속도 출력)
edge/benchmark.py  카메라 없이 모델 추론 속도 재기
server/            웹 알림 개발 서버 + 대시보드
configs/           데이터셋·학습·실험·엣지 설정
docs/api.md        엣지·서버·웹 사이 요청 형식
```

## 참고

- 데이터셋 인용
  ```
  @misc{ baby-position-esrva_dataset,
    title = { Baby-Position Dataset }, author = { Jennifers Workspace },
    howpublished = { \url{ https://universe.roboflow.com/jennifer-verona-uzduw/baby-position-esrva } },
    journal = { Roboflow Universe }, publisher = { Roboflow }, year = { 2026 }, month = { oct } }
  ```
- G. L. Pirotello, A. T. Akabane, D. C. González, "A Lightweight Deep Learning Approach for Autonomous Detection of Risky Infant Sleep Postures," SBCUP 2026. https://sol.sbc.org.br/index.php/sbcup/article/view/43282
- Ultralytics YOLO: https://docs.ultralytics.com
- 미국소아과학회(AAP) 안전한 수면 가이드: https://www.healthychildren.org/English/ages-stages/baby/sleep/Pages/a-parents-guide-to-safe-sleep.aspx

## 라이선스

코드는 [MIT](LICENSE)예요. 데이터셋은 원 저작자의 CC BY 4.0을 따르고, 쓸 때 위 인용을 표기해야 해요.
