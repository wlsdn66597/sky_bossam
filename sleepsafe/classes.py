"""모델 클래스 정의. configs/dataset.yaml의 classes 순서와 같아야 해요."""

CLASSES = ("supine", "side", "prone", "airway_visible", "airway_covered")
BODY_CLASSES = ("supine", "side", "prone")          # 몸 전체 박스
AIRWAY_CLASSES = ("airway_visible", "airway_covered")  # 얼굴(코·입) 박스

KOREAN = {
    "supine": "바로 누움",
    "side": "옆으로 누움",
    "prone": "엎드림",
    "airway_visible": "코·입 보임",
    "airway_covered": "코·입 가림",
    "face_covered": "얼굴 가림",
    "normal": "정상",
    "caution": "주의",
    "risk": "위험",
    "unavailable": "분석 불가",
    "no_baby": "아기 안 보임",
    "dark": "너무 어두움",
    "body_not_found": "몸은 안 보이고 가려진 얼굴만 보임",
}
