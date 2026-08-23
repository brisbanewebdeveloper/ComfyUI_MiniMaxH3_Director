"""Bounded-memory temporal and spatial sampling for Director Refine."""

from __future__ import annotations

from typing import Any, Callable

import torch

from .h3_context_patches import CTX_FRAME_KEY
from .h3_motion_context import FRAME_PER_TOKEN, FRAME_RESCALE, pixel_frames_for_latent_t

SamplePiece = Callable[[dict, Any, Any, int], dict]
ProgressCallback = Callable[[int, int], None]


def _streams(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(value, "unbind"):
        parts = list(value.unbind())
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        raise ValueError(f"Tiled Refine expects a MiniMax H3 AV latent, got {type(value)!r}.")
    if len(parts) != 2:
        raise ValueError(f"Tiled Refine expects two AV streams, got {len(parts)}.")
    return parts[0], parts[1]


def _nested_like(template: Any, video: torch.Tensor, audio: torch.Tensor) -> Any:
    if hasattr(template, "tensors"):
        return type(template)((video, audio))
    if isinstance(template, tuple):
        return (video, audio)
    if isinstance(template, list):
        return [video, audio]
    raise ValueError(f"Unsupported MiniMax H3 AV container: {type(template)!r}.")


def _frames_before_token(token: int) -> int:
    cycles, remainder = divmod(max(0, int(token)), len(FRAME_PER_TOKEN))
    return cycles * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:remainder])


def temporal_windows(video_tokens: int, chunk_length: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Return cycle-aligned ``(token_start, token_end, frame_start, frame_end)`` windows."""
    chunk_length, overlap = int(chunk_length), int(overlap)
    if chunk_length % 17 or overlap % 17:
        raise ValueError("Tiled Refine temporal chunk length and overlap must be multiples of 17 frames.")
    if chunk_length <= overlap:
        raise ValueError("Tiled Refine temporal overlap must be smaller than chunk length.")
    chunk_tokens = chunk_length // 17 * 5
    overlap_tokens = overlap // 17 * 5
    hop = chunk_tokens - overlap_tokens
    total = max(1, int(video_tokens))
    windows = []
    start = 0
    while start < total:
        end = min(total, start + chunk_tokens)
        windows.append((start, end, _frames_before_token(start), _frames_before_token(end)))
        if end >= total:
            break
        start += hop
    return windows


def _grid_axis(size: int, tile: int, overlap: int, minimum: int) -> list[tuple[int, int, int]]:
    if tile <= 0 or overlap >= tile:
        raise ValueError("Tiled Refine tile overlap must be smaller than each tile dimension.")
    if minimum > tile:
        raise ValueError("Tiled Refine minimum tile size must not exceed tile width or height.")
    if size <= tile:
        return [(0, size, 0)]
    positions = []
    start = 0
    while start < size:
        end = min(size, start + tile)
        if end - start < minimum and positions:
            start = max(positions[-1][0] + 1, size - minimum)
            end = size
        previous_end = positions[-1][1] if positions else start
        positions.append((start, end, max(0, previous_end - start)))
        if end >= size:
            break
        start += tile - overlap
    return positions


def spatial_grid(height: int, width: int, settings: dict[str, Any]) -> list[tuple[int, int, int, int, int, int]]:
    """Return latent-space tile bounds and actual top/left overlaps."""
    for key in ("tiled_refine_tile_width", "tiled_refine_tile_height", "tiled_refine_tile_overlap", "tiled_refine_fade", "tiled_refine_min_tile_size"):
        if int(settings[key]) % 32:
            raise ValueError(f"{key} must be a multiple of 32 pixels.")
    tile_w = int(settings["tiled_refine_tile_width"]) // 16
    tile_h = int(settings["tiled_refine_tile_height"]) // 16
    overlap = int(settings["tiled_refine_tile_overlap"]) // 16
    minimum = int(settings["tiled_refine_min_tile_size"]) // 16
    rows = _grid_axis(int(height), tile_h, overlap, minimum)
    cols = _grid_axis(int(width), tile_w, overlap, minimum)
    return [(r0, r1, c0, c1, top, left) for r0, r1, top in rows for c0, c1, left in cols]


def _slice_keyframe(keyframe: dict[str, Any], frame_start: int, frame_end: int) -> dict[str, Any] | None:
    item = dict(keyframe)
    index = int(item.get(CTX_FRAME_KEY, item.get("resolved_frame_index", 0)))
    video = item.get("latent")
    video_frames = pixel_frames_for_latent_t(int(video.shape[2])) if torch.is_tensor(video) else 1
    audio = item.get("audio_latent")
    audio_frames = int(audio.shape[-1] / FRAME_RESCALE) if torch.is_tensor(audio) else 1
    span = max(video_frames, audio_frames, 1)
    if index < frame_start or index + span > frame_end:
        return None
    relative = index - frame_start
    item["resolved_frame_index"] = relative
    if CTX_FRAME_KEY in item:
        item[CTX_FRAME_KEY] = relative
    return item


def conditioning_for_piece(
    conditioning: Any,
    *,
    frame_start: int,
    frame_end: int,
    source_height: int,
    source_width: int,
    tile: tuple[int, int, int, int] | None = None,
) -> Any:
    """Rebase H3 keyframes to a temporal window and crop target-sized keyframes to a tile."""
    if not conditioning:
        return conditioning
    out = []
    for cond, metadata in conditioning:
        values = dict(metadata)
        if "minimax_keyframes" in values:
            keyframes = []
            for raw in values.get("minimax_keyframes") or ():
                item = _slice_keyframe(raw, frame_start, frame_end)
                if item is None:
                    continue
                latent = item.get("latent")
                if tile is not None and torch.is_tensor(latent) and tuple(latent.shape[-2:]) == (source_height, source_width):
                    r0, r1, c0, c1 = tile
                    item["latent"] = latent[..., r0:r1, c0:c1].contiguous()
                keyframes.append(item)
            values["minimax_keyframes"] = keyframes
        values["minimax_frame_count"] = max(1, frame_end - frame_start)
        out.append([cond, values])
    return out


def _curve(length: int, blend: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if blend == "overwrite":
        return torch.ones(length, device=device, dtype=dtype)
    values = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
    return values * values * (3.0 - 2.0 * values) if blend == "smoothstep" else values


def _piece_latent(
    template: dict,
    video: torch.Tensor,
    audio: torch.Tensor,
    video_mask: torch.Tensor,
) -> dict:
    out = dict(template)
    out["samples"] = _nested_like(template["samples"], video, audio)
    mask_template = template.get("noise_mask", template["samples"])
    out["noise_mask"] = _nested_like(mask_template, video_mask, torch.zeros_like(audio))
    return out


def sample_tiled_refine(
    latent: dict,
    positive: Any,
    negative: Any,
    settings: dict[str, Any],
    *,
    seed: int,
    sample_piece: SamplePiece,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Sample an H3 AV latent one bounded temporal/spatial piece at a time."""
    sample_template = latent["samples"]
    video, audio = _streams(sample_template)
    mask_value = latent.get("noise_mask")
    if mask_value is None:
        video_mask = torch.ones((video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]), device=video.device, dtype=video.dtype)
    else:
        video_mask, _ = _streams(mask_value)
        if video_mask.ndim != 5:
            raise ValueError(f"Tiled Refine video noise mask must be 5D, got {tuple(video_mask.shape)}.")
        video_mask = video_mask.expand(
            video.shape[0],
            video_mask.shape[1],
            video.shape[2],
            video.shape[3],
            video.shape[4],
        )
    if settings["tiled_refine_temporal"]:
        windows = temporal_windows(video.shape[2], settings["tiled_refine_chunk_length"], settings["tiled_refine_temporal_overlap"])
    else:
        windows = [(0, video.shape[2], 0, _frames_before_token(video.shape[2]))]
    grid = spatial_grid(video.shape[3], video.shape[4], settings) if settings["tiled_refine_spatial"] else [(0, video.shape[3], 0, video.shape[4], 0, 0)]
    total = len(windows) * len(grid)
    completed = 0
    result = video.clone()
    blend = settings["tiled_refine_blend"]
    earlier = settings["tiled_refine_overlap_mode"] == "earlier"
    fade = int(settings["tiled_refine_fade"]) // 16

    for window_index, (k0, k1, f0, f1) in enumerate(windows):
        chunk = video[:, :, k0:k1].clone()
        chunk_mask = video_mask[:, :, k0:k1].clone()
        temporal_overlap = max(0, windows[window_index - 1][1] - k0) if window_index else 0
        if temporal_overlap:
            chunk[:, :, :temporal_overlap] = result[:, :, k0:k0 + temporal_overlap]
            ramp = _curve(temporal_overlap, blend, device=chunk.device, dtype=chunk_mask.dtype)
            if earlier:
                chunk_mask[:, :, :temporal_overlap] *= ramp[None, None, :, None, None]
        chunk_result = chunk.clone()
        a0, a1 = round(f0 * FRAME_RESCALE), min(audio.shape[-1], round(f1 * FRAME_RESCALE))
        chunk_audio = audio[..., a0:a1].contiguous()

        for r0, r1, c0, c1, top, left in grid:
            tile = chunk_result[..., r0:r1, c0:c1].clone()
            tile_mask = chunk_mask[..., r0:r1, c0:c1].clone()
            if earlier and left:
                width = min(left, fade) if fade else left
                ramp = _curve(width, blend, device=tile.device, dtype=tile_mask.dtype)
                tile_mask[..., :width] *= ramp[None, None, None, None, :]
            if earlier and top:
                height = min(top, fade) if fade else top
                ramp = _curve(height, blend, device=tile.device, dtype=tile_mask.dtype)
                tile_mask[..., :height, :] *= ramp[None, None, None, :, None]
            tile_bounds = (r0, r1, c0, c1)
            cond = conditioning_for_piece(positive, frame_start=f0, frame_end=f1, source_height=video.shape[3], source_width=video.shape[4], tile=tile_bounds)
            neg = conditioning_for_piece(negative, frame_start=f0, frame_end=f1, source_height=video.shape[3], source_width=video.shape[4], tile=tile_bounds)
            piece = _piece_latent(latent, tile, chunk_audio, tile_mask)
            sampled = sample_piece(piece, cond, neg, int(seed) + completed)
            sampled_video, _ = _streams(sampled["samples"])
            old = chunk_result[..., r0:r1, c0:c1]
            weight = torch.ones((r1 - r0, c1 - c0), device=old.device, dtype=old.dtype)
            if earlier and left:
                weight[:, :left] *= _curve(left, blend, device=old.device, dtype=old.dtype)[None, :]
            if earlier and top:
                weight[:top, :] *= _curve(top, blend, device=old.device, dtype=old.dtype)[:, None]
            chunk_result[..., r0:r1, c0:c1] = old * (1.0 - weight[None, None, None]) + sampled_video * weight[None, None, None]
            completed += 1
            if on_progress is not None:
                on_progress(completed, total)

        if temporal_overlap and earlier:
            weight = _curve(temporal_overlap, blend, device=result.device, dtype=result.dtype)
            result[:, :, k0:k0 + temporal_overlap] = result[:, :, k0:k0 + temporal_overlap] * (1.0 - weight[None, None, :, None, None]) + chunk_result[:, :, :temporal_overlap] * weight[None, None, :, None, None]
            result[:, :, k0 + temporal_overlap:k1] = chunk_result[:, :, temporal_overlap:]
        else:
            result[:, :, k0:k1] = chunk_result

    out = dict(latent)
    out.pop("noise_mask", None)
    out["samples"] = _nested_like(sample_template, result, audio)
    return out
