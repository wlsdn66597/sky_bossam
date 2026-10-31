"""설정 파일 읽기와 경로 처리."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

ROOT = Path(__file__).resolve().parents[1]  # 레포 최상위 폴더


def repo_path(p: Union[str, Path]) -> Path:
    """상대 경로는 레포 최상위 기준으로 바꿔요."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    return yaml.safe_load(repo_path(path).read_text(encoding="utf-8")) or {}


def parse_overrides(items) -> Dict[str, Any]:
    """['epochs=50', 'device=cpu'] → {'epochs': 50, 'device': 'cpu'}"""
    out: Dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"key=value 형식이어야 해요: {item}")
        out[key.strip()] = yaml.safe_load(value)
    return out
