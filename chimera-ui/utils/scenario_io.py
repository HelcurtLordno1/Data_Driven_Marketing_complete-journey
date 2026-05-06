"""
chimera-ui/utils/scenario_io.py
Save and load named weight-scenario JSON files for the Scenario Manager.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

_SCENARIO_DIR = Path(__file__).resolve().parent.parent / "data"
_SCENARIO_FILE = _SCENARIO_DIR / "saved_scenarios.json"


def _load_raw() -> List[Dict]:
    if _SCENARIO_FILE.exists():
        try:
            return json.loads(_SCENARIO_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_raw(data: List[Dict]) -> None:
    _SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    _SCENARIO_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_scenario_names() -> List[str]:
    return [s["name"] for s in _load_raw()]


def save_scenario_to_disk(
    name: str,
    weights: Dict[str, float],
    archetype_filter: Optional[str] = None,
    description: str = "",
) -> None:
    data = _load_raw()
    data = [s for s in data if s.get("name") != name]  # overwrite
    data.append({
        "name":            name,
        "weights":         weights,
        "archetype_filter": archetype_filter,
        "description":     description,
    })
    _save_raw(data)


def load_scenario_from_disk(name: str) -> Optional[Dict]:
    for s in _load_raw():
        if s.get("name") == name:
            return s
    return None


def delete_scenario(name: str) -> None:
    data = [s for s in _load_raw() if s.get("name") != name]
    _save_raw(data)
