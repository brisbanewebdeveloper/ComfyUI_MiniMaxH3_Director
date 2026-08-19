"""Resolve which MiniMax H3 checkpoint family a queued timeline needs."""

from __future__ import annotations

import json
from typing import Any

from ..lib.task_prompts import resolve_task_key


REF2VA_TASK_KEYS = frozenset({"r2v", "v2v", "rv2v"})
FL2VA_FAMILY = "fl2va"
REF2VA_FAMILY = "ref2va"


def model_family_for_task(task_type: str) -> str:
    return REF2VA_FAMILY if resolve_task_key(str(task_type or "")) in REF2VA_TASK_KEYS else FL2VA_FAMILY


def timeline_model_families(timeline_data: str, task_type: str = "") -> frozenset[str]:
    """Return checkpoint families used by selected segments in a timeline."""
    try:
        timeline = json.loads(timeline_data) if str(timeline_data or "").strip() else {}
    except (TypeError, json.JSONDecodeError):
        timeline = {}
    if not isinstance(timeline, dict):
        timeline = {}

    global_task = str((timeline.get("global") or {}).get("taskType") or task_type or "t2v")
    segments = timeline.get("segments") or timeline.get("shots") or []
    if not isinstance(segments, list) or not segments:
        return frozenset({model_family_for_task(global_task)})

    indices = range(len(segments))
    run_enabled = bool(timeline.get("runSelectEnabled") or timeline.get("run_select_enabled"))
    raw_selection = timeline.get("runSelection", timeline.get("run_selection"))
    if run_enabled and isinstance(raw_selection, list) and raw_selection:
        selected = {int(value) for value in raw_selection if str(value).lstrip("-").isdigit()}
        indices = (index for index in range(len(segments)) if index in selected)

    families = {
        model_family_for_task(
            str(segment.get("taskType") or segment.get("task_type") or global_task)
            if isinstance(segment, dict)
            else global_task
        )
        for index in indices
        for segment in [segments[index]]
    }
    return frozenset(families or {model_family_for_task(global_task)})


def model_for_segment(model: Any, model_ref2va: Any, task_key: str) -> Any:
    if model_family_for_task(task_key) == REF2VA_FAMILY:
        preferred, fallback = model_ref2va, model
    else:
        preferred, fallback = model, model_ref2va
    if preferred is not None:
        return preferred
    if fallback is not None:
        return fallback
    raise ValueError(
        "MiniMax H3 Director: no model connected. Connect FL2VA to model and/or "
        "REF2VA to model_ref2va."
    )
