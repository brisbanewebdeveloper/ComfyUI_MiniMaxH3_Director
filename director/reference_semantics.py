"""Deterministic MiniMax H3 reference metadata and prompt compilation."""

from __future__ import annotations

import re
from typing import Any


IMAGE_RETENTION = ("fully_preserved", "partially_preserved", "reference", "weak_reference")
AUDIO_RETENTION = ("fully_copy", "partially_copy", "reference", "weak_reference")
DEFAULT_KIND = "subject"

_DIALOGUE_RE = re.compile(
    r"^(?P<indent>\s*)<(?:Subject|Picture)\s+(?P<index>\d+)\s*>\s+says\s*:\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)


def _index(item: dict[str, Any], fallback: int) -> int:
    try:
        return max(0, int(item.get("index", item.get("slot", fallback))))
    except (TypeError, ValueError):
        return fallback


def _text(item: dict[str, Any], *names: str) -> str:
    for name in names:
        value = item.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _semantic_item(item: Any, *, media: str, fallback: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    idx = _index(item, fallback)
    retention_values = AUDIO_RETENTION if media == "audio" else IMAGE_RETENTION
    retention = _text(item, "retention")
    if retention not in retention_values:
        retention = "reference"
    out = {
        "index": idx,
        "media": media,
        "kind": _text(item, "subjectKind", "subject_kind", "kind") or DEFAULT_KIND,
        "description": _text(item, "description", "describes"),
        "retention": retention,
        "retained": _text(item, "retained", "retentionDescription", "retention_description"),
        "voice_of": _text(item, "voiceOf", "voice_of"),
    }
    if not any(out[name] for name in ("description", "retained", "voice_of")) and not any(
        name in item for name in ("subjectKind", "subject_kind", "kind", "retention")
    ):
        return None
    return out


def collect_reference_semantics(
    refs: list[Any] | None,
    videos: list[Any] | None,
    audios: list[Any] | None,
) -> list[dict[str, Any]]:
    """Return normalized semantic records without requiring attached media tensors."""
    records: list[dict[str, Any]] = []
    for media, items in (("image", refs), ("video", videos), ("audio", audios)):
        for fallback, item in enumerate(items or []):
            normalized = _semantic_item(item, media=media, fallback=fallback)
            if normalized is not None:
                records.append(normalized)
    return records


def merge_reference_semantics(common: list[dict[str, Any]], local: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge common and local records by media and slot; local metadata wins."""
    merged = {(item["media"], item["index"]): item for item in common}
    merged.update({(item["media"], item["index"]): item for item in local})
    return [merged[key] for key in sorted(merged, key=lambda value: (value[0], value[1]))]


def reference_semantics_for_segment(plan, seg) -> list[dict[str, Any]]:
    """Read additive reference metadata from a Director plan's serialized timeline."""
    raw = plan.raw or {}
    global_block = raw.get("global") or {}
    rows = raw.get("segments") or []
    ui_index = seg.ui_index if seg.ui_index is not None else seg.index
    row = rows[ui_index] if 0 <= ui_index < len(rows) and isinstance(rows[ui_index], dict) else {}

    def collect(block: dict[str, Any]) -> list[dict[str, Any]]:
        videos = block.get("refVideos") or block.get("ref_videos") or []
        legacy_video = block.get("referenceVideo") or block.get("reference_video") or {}
        if isinstance(legacy_video, dict) and legacy_video and not videos:
            videos = [{"index": 0, **legacy_video}]
        return collect_reference_semantics(
            block.get("refs") or [],
            videos,
            block.get("refAudios") or block.get("ref_audios") or [],
        )

    common = collect(global_block)
    if seg.use_global:
        return common
    local = collect(row)
    common_enabled = bool(global_block.get("commonEnabled", global_block.get("common_enabled", False)))
    if seg.task_key == "r2v" and common_enabled:
        return merge_reference_semantics(common, local)
    return local


def _speaker_subject(value: str) -> int | None:
    match = re.search(r"(?:Subject|Picture)?\s*(\d+)", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _dialogue_prompt(prompt: str, speaker_ids: dict[int, int]) -> str:
    lines: list[str] = []
    for line in prompt.splitlines():
        match = _DIALOGUE_RE.match(line)
        if not match:
            lines.append(line)
            continue
        subject = int(match.group("index"))
        speaker = speaker_ids.setdefault(subject, len(speaker_ids) + 1)
        lines.append(
            f"{match.group('indent')}<Subject {subject}> (S{speaker}): <d>{match.group('text').strip()}</d>"
        )
    return "\n".join(lines)


def compile_reference_prompt(prompt: str, records: list[dict[str, Any]]) -> str:
    """Add structured sections only when metadata exists; preserve manual sections."""
    if not records:
        return prompt

    speaker_ids: dict[int, int] = {}
    body = _dialogue_prompt(prompt or "", speaker_ids)
    definitions: list[str] = []
    retention_lines: list[str] = []
    voice_lines: list[str] = []

    for item in records:
        number = int(item["index"]) + 1
        media = item["media"]
        if media == "image":
            description = item["description"] or f"the {item['kind']} shown in <Picture {number}>"
            definitions.append(f"<Subject {number}> is {description}.")
            label = f"<Subject {number}>"
        elif media == "video":
            if item["description"]:
                definitions.append(f"<Video {number}> is {item['description']}.")
            label = f"<Video {number}>"
        else:
            if item["description"]:
                definitions.append(f"<Audio {number}> is {item['description']}.")
            label = f"<Audio {number}>"

        retained = item["retained"] or f"the referenced {item['kind']} remains recognizable"
        retention_lines.append(f"{label}: {item['retention']} - {retained}.")
        if media == "audio" and item["voice_of"]:
            subject = _speaker_subject(item["voice_of"])
            if subject is not None:
                speaker = speaker_ids.setdefault(subject, len(speaker_ids) + 1)
                voice_lines.append(f"<Audio {number}> is a voice reference for <Subject {subject}> (S{speaker}).")

    sections: list[str] = []
    lower = body.lower()
    definition_lines = definitions + voice_lines
    if definition_lines and "subject_definitions:" not in lower:
        sections.append("subject_definitions:\n" + "\n".join(definition_lines))
    if retention_lines and "retention_analysis:" not in lower:
        sections.append("retention_analysis:\n" + "\n".join(retention_lines))
    sections.append(body.strip())
    return "\n\n".join(section for section in sections if section)
