"""External Refine pack for MiniMax H3 Director (graph-wired config).

Connect ``MiniMaxH3DirectorRefine.refine`` → ``MiniMaxH3Director.refine``.
Unconnected = current single-pass sampling.
"""

from __future__ import annotations

import math
from typing import Any

from ..lib.image_prep import MINIMAX_CANVAS_STRIDE, ensure_minimax_canvas

MMX_DIR_REFINE = "MMX_DIR_REFINE"

REFINE_MODES = ("refine", "upscale", "latent_upscale")
SEED_MODES = ("inherit", "offset", "fixed")
UPSCALE_METHODS = ("lanczos", "nvidia_rtx_vsr", "h3_latent")
LATENT_UPSCALE_DEVICES = ("auto", "cuda", "cpu")
LATENT_UPSCALE_PRECISIONS = ("fp16", "fp32", "bf16")
SIGMA_SPACINGS = ("linear", "cosine", "sine")
MAX_REFINE_PASSES = 9999
# 海螺参考生视频二采：ManualSigmas 4 个数 = euler 3 步。
HAILUO_REFINE_SIGMAS = (0.85, 0.7250, 0.4219, 0.0)
DEFAULT_REFINE_SIGMA_SAMPLER = "euler"


def parse_refine_sigmas(raw: Any, *, fallback: bool = False) -> tuple[float, ...]:
    """Parse ManualSigmas text or a BasicScheduler SIGMAS tensor."""
    vals: list[float] = []
    from_tensor = is_refine_sigmas_tensor(raw)
    if from_tensor:
        try:
            vals = [float(x) for x in raw.detach().float().cpu().reshape(-1).tolist()]
        except Exception:
            vals = []
    elif isinstance(raw, (list, tuple)):
        for part in raw:
            try:
                vals.append(float(part))
            except (TypeError, ValueError):
                continue
    else:
        text = str(raw or "").replace(";", ",").replace("\n", ",")
        for part in text.split(","):
            token = str(part).strip()
            if not token:
                continue
            try:
                vals.append(float(token))
            except (TypeError, ValueError):
                continue
    if len(vals) < 2:
        if not fallback:
            raise ValueError(
                "Refine SIGMAS 至少需要 2 个数（步数 + 结尾 0）。"
                "请检查 BasicScheduler 的 steps / denoise。"
            )
        return HAILUO_REFINE_SIGMAS
    if abs(vals[-1]) > 1e-8:
        vals.append(0.0)
    return tuple(vals)


def is_refine_sigmas_tensor(raw: Any) -> bool:
    return raw is not None and not isinstance(raw, (str, bytes, list, tuple)) and hasattr(raw, "reshape")


def refine_sigmas_override(pack: dict[str, Any] | None):
    """Return wired SIGMAS, otherwise the generated second-pass schedule."""
    pack = pack or {}
    tensor = pack.get("sigmas_tensor")
    if tensor is None and is_refine_sigmas_tensor(pack.get("sigmas")):
        tensor = pack.get("sigmas")
    if tensor is not None:
        return parse_refine_sigmas(tensor, fallback=False)
    parsed = pack.get("sigmas_parsed") or ()
    return parse_refine_sigmas(parsed, fallback=False) if parsed else None


def generated_refine_sigmas(steps: int, first_sigma: float, spacing: str) -> tuple[float, ...]:
    """Build the SetFirstSigma + ExtendIntermediateSigmas schedule used by the example."""
    steps = max(1, min(100, int(steps or 1)))
    first = max(0.0, min(20000.0, float(first_sigma)))
    spacing = str(spacing or "linear").strip().lower()
    if spacing not in SIGMA_SPACINGS:
        spacing = "linear"
    values = [first]
    for index in range(1, steps):
        x = index / steps
        if spacing == "cosine":
            x = math.sin(x * math.pi / 2)
        elif spacing == "sine":
            x = 1 - math.cos(x * math.pi / 2)
        values.append(first * (1 - x))
    values.append(0.0)
    return tuple(values)


def resolve_latent_upscale_ref(raw: Any) -> tuple[Any, str]:
    """Return (loaded_module_or_None, filename)."""
    if raw is None or raw is False:
        return None, ""
    if isinstance(raw, str):
        name = raw.strip()
        return None, "" if name.startswith("(") else name
    if isinstance(raw, dict):
        name = str(raw.get("name") or raw.get("model_name") or "").strip()
        if name.startswith("("):
            name = ""
        return raw.get("model"), name
    name = str(getattr(raw, "name", None) or getattr(raw, "_h3_name", None) or "").strip()
    if hasattr(raw, "parameters") or hasattr(raw, "model"):
        return raw, name or type(raw).__name__
    return None, name


def refine_needs_canvas(pack: dict[str, Any] | None) -> bool:
    mode = str((pack or {}).get("mode") or "").strip().lower()
    return mode in {"upscale", "latent_upscale"}


def refine_uses_h3_latent(pack: dict[str, Any] | None) -> bool:
    pack = pack or {}
    mode = str(pack.get("mode") or "").strip().lower()
    if mode == "latent_upscale":
        return True
    method = str(pack.get("upscale_method") or "").strip().lower()
    return mode == "upscale" and method == "h3_latent"


def latent_upscale_model_name(pack: dict[str, Any] | None) -> str:
    pack = pack or {}
    _model, name = resolve_latent_upscale_ref(
        pack.get("latent_upscale_ref")
        if pack.get("latent_upscale_ref") is not None
        else pack.get("latent_upscale_model")
    )
    if name:
        return name
    return str(pack.get("h3_latent_model") or "").strip()


FOLLOW_DIRECTOR_ASPECT = "Follow Director"
SCALE_BY_ASPECT = "Scale by multiplier"
CUSTOM_ASPECT_RATIO = "Custom"
DEFAULT_UPSCALE_MEGAPIXELS = 1.0

# English values keep the Refine node portable; the ratio math matches Director.
RESOLUTION_ASPECTS = (
    ("1:1 (Square)", 1, 1),
    ("2:3 (Portrait photo)", 2, 3),
    ("3:2 (Landscape photo)", 3, 2),
    ("3:4 (Portrait standard)", 3, 4),
    ("4:3 (Standard)", 4, 3),
    ("9:16 (Portrait)", 9, 16),
    ("16:9 (Widescreen)", 16, 9),
    ("21:9 (Ultrawide)", 21, 9),
)

ASPECT_RATIO_CHOICES = (
    FOLLOW_DIRECTOR_ASPECT,
    SCALE_BY_ASPECT,
    *[row[0] for row in RESOLUTION_ASPECTS],
    CUSTOM_ASPECT_RATIO,
)

_ASPECT_ALIASES = {
    "跟随导演台": FOLLOW_DIRECTOR_ASPECT,
    "follow": FOLLOW_DIRECTOR_ASPECT,
    "按倍数": SCALE_BY_ASPECT,
    "scale_by": SCALE_BY_ASPECT,
    "自定义": CUSTOM_ASPECT_RATIO,
    "自定义 (Custom)": CUSTOM_ASPECT_RATIO,
    "1:1 (方形)": "1:1 (Square)",
    "2:3 (竖版照片)": "2:3 (Portrait photo)",
    "3:2 (横版照片)": "3:2 (Landscape photo)",
    "3:4 (竖版标准)": "3:4 (Portrait standard)",
    "4:3 (标准)": "4:3 (Standard)",
    "9:16 (竖屏)": "9:16 (Portrait)",
    "16:9 (宽屏)": "16:9 (Widescreen)",
    "21:9 (超宽)": "21:9 (Ultrawide)",
    "1:1 (Square)": "1:1 (Square)",
    "2:3 (Portrait Photo)": "2:3 (Portrait photo)",
    "3:2 (Photo)": "3:2 (Landscape photo)",
    "3:4 (Portrait Standard)": "3:4 (Portrait standard)",
    "4:3 (Standard)": "4:3 (Standard)",
    "9:16 (Portrait Widescreen)": "9:16 (Portrait)",
    "16:9 (Widescreen)": "16:9 (Widescreen)",
    "21:9 (Ultrawide)": "21:9 (Ultrawide)",
}


def infer_upscale_target(base_w: int, base_h: int) -> tuple[int, int]:
    """Default 720p-class canvas from a 480p-class source (snap to H3 ×32)."""
    w, h = int(base_w or 0), int(base_h or 0)
    if w <= 0 or h <= 0:
        return ensure_minimax_canvas(1280, 720)
    if w >= h:
        nh = 720
        nw = max(32, round(w * nh / h))
        return ensure_minimax_canvas(nw, nh)
    nw = 720
    nh = max(32, round(h * nw / w))
    return ensure_minimax_canvas(nw, nh)


def normalize_aspect_ratio(aspect_ratio: str | None) -> str:
    # Legacy workflows mapped old target_width=0 onto this combo.
    if aspect_ratio in (None, "", 0, 0.0, "0", "0.0", False):
        return FOLLOW_DIRECTOR_ASPECT
    v = str(aspect_ratio).strip()
    if not v or v in {"0", "0.0", "None", "null"}:
        return FOLLOW_DIRECTOR_ASPECT
    if v in _ASPECT_ALIASES:
        return _ASPECT_ALIASES[v]
    if v in ASPECT_RATIO_CHOICES:
        return v
    prefix = v.split(" ", 1)[0]
    for label, _, _ in RESOLUTION_ASPECTS:
        if label == prefix or label.startswith(f"{prefix} "):
            return label
    if v.startswith("自定义") or v.lower() == "custom":
        return CUSTOM_ASPECT_RATIO
    return FOLLOW_DIRECTOR_ASPECT


def is_follow_director_aspect(aspect_ratio: str | None) -> bool:
    return normalize_aspect_ratio(aspect_ratio) == FOLLOW_DIRECTOR_ASPECT


def is_custom_aspect_ratio(aspect_ratio: str | None) -> bool:
    return normalize_aspect_ratio(aspect_ratio) == CUSTOM_ASPECT_RATIO


def is_scale_by_aspect(aspect_ratio: str | None) -> bool:
    return normalize_aspect_ratio(aspect_ratio) == SCALE_BY_ASPECT


def resolution_from_selector(
    aspect_ratio: str,
    megapixels: float,
    multiple: int = MINIMAX_CANVAS_STRIDE,
) -> tuple[int, int] | None:
    """Director ResolutionSelector math: aspect + MP → W×H snapped to ×32."""
    ar = normalize_aspect_ratio(aspect_ratio)
    if ar in {FOLLOW_DIRECTOR_ASPECT, SCALE_BY_ASPECT, CUSTOM_ASPECT_RATIO}:
        return None
    row = next((r for r in RESOLUTION_ASPECTS if r[0] == ar), None)
    if row is None:
        return None
    _, w_ratio, h_ratio = row
    try:
        mp = float(megapixels)
    except (TypeError, ValueError):
        mp = DEFAULT_UPSCALE_MEGAPIXELS
    mp = min(16.0, max(0.1, mp))
    mult = max(8, int(multiple or MINIMAX_CANVAS_STRIDE))
    scale = math.sqrt((mp * 1024 * 1024) / (w_ratio * h_ratio))
    width = int(round((w_ratio * scale) / mult) * mult)
    height = int(round((h_ratio * scale) / mult) * mult)
    return ensure_minimax_canvas(max(width, mult), max(height, mult))


def resolve_refine_target(
    *,
    aspect_ratio: str = FOLLOW_DIRECTOR_ASPECT,
    megapixels: float = DEFAULT_UPSCALE_MEGAPIXELS,
    width: int = 0,
    height: int = 0,
    target_width: int = 0,
    target_height: int = 0,
) -> tuple[int, int]:
    """Return (0, 0) to follow Director canvas; otherwise an explicit ×32 canvas."""
    ar = normalize_aspect_ratio(aspect_ratio)
    legacy_w, legacy_h = int(target_width or 0), int(target_height or 0)
    w, h = int(width or 0), int(height or 0)
    if is_follow_director_aspect(ar):
        if (legacy_w > 0 or legacy_h > 0) and w <= 0 and h <= 0:
            return ensure_minimax_canvas(max(legacy_w, 32), max(legacy_h, 32))
        return 0, 0
    if is_custom_aspect_ratio(ar):
        cw = w or legacy_w or 1280
        ch = h or legacy_h or 720
        return ensure_minimax_canvas(max(cw, 32), max(ch, 32))
    resolved = resolution_from_selector(ar, megapixels)
    if resolved is not None:
        return resolved
    return 0, 0


def pack_refine(
    *,
    mode: str = "refine",
    passes: int = 1,
    seed_mode: str = "inherit",
    aspect_ratio: str = FOLLOW_DIRECTOR_ASPECT,
    megapixels: float = DEFAULT_UPSCALE_MEGAPIXELS,
    width: int = 0,
    height: int = 0,
    target_width: int = 0,
    target_height: int = 0,
    skip_fl2v: bool = True,
    scale_by: float = 1.5,
    latent_upscale_device: str = "auto",
    latent_upscale_precision: str = "fp16",
    strict: bool = True,
    second_pass_steps: int = 2,
    first_sigma: float = 0.8,
    sigma_spacing: str = "linear",
    second_seed: int = 0,
    upscale_method: str = "h3_latent",
    sample_model=None,
    latent_upscale_model=None,
    upscale_model=None,
    sampler: str = "",
    sigmas=None,
) -> dict[str, Any]:
    mode = str(mode or "refine").strip().lower()
    if mode not in REFINE_MODES:
        mode = "refine"
    seed_mode = str(seed_mode or "inherit").strip().lower()
    if seed_mode not in SEED_MODES:
        seed_mode = "inherit"
    method = str(upscale_method or "h3_latent").strip().lower()
    if method not in UPSCALE_METHODS:
        method = "h3_latent"
    sampler = str(sampler or DEFAULT_REFINE_SIGMA_SAMPLER).strip() or DEFAULT_REFINE_SIGMA_SAMPLER
    sigma_tensor = sigmas if is_refine_sigmas_tensor(sigmas) else None
    sigma_spacing = str(sigma_spacing or "linear").strip().lower()
    if sigma_spacing not in SIGMA_SPACINGS:
        sigma_spacing = "linear"
    parsed = parse_refine_sigmas(sigma_tensor, fallback=False) if sigma_tensor is not None else ()
    if sigma_tensor is None and mode != "latent_upscale":
        parsed = generated_refine_sigmas(second_pass_steps, first_sigma, sigma_spacing)
    ar = normalize_aspect_ratio(aspect_ratio)
    tw, th = resolve_refine_target(
        aspect_ratio=ar,
        megapixels=megapixels,
        width=width,
        height=height,
        target_width=target_width,
        target_height=target_height,
    )
    latent_mod, latent_name = resolve_latent_upscale_ref(latent_upscale_model)
    device = str(latent_upscale_device or "auto").strip().lower()
    if device not in LATENT_UPSCALE_DEVICES:
        device = "auto"
    precision = str(latent_upscale_precision or "fp16").strip().lower()
    if precision not in LATENT_UPSCALE_PRECISIONS:
        precision = "fp16"
    return {
        "enabled": True,
        "mode": mode,
        "passes": refine_passes_for({"passes": passes}),
        "seed_mode": seed_mode,
        "aspect_ratio": ar,
        "megapixels": float(megapixels or DEFAULT_UPSCALE_MEGAPIXELS),
        "target_width": tw,
        "target_height": th,
        "skip_fl2v": bool(skip_fl2v),
        "scale_by": max(1.0, min(4.0, float(scale_by or 1.5))),
        "latent_upscale_device": device,
        "latent_upscale_precision": precision,
        "strict": bool(strict),
        "second_pass_steps": max(1, min(100, int(second_pass_steps or 1))),
        "first_sigma": max(0.0, min(20000.0, float(first_sigma))),
        "sigma_spacing": sigma_spacing,
        "second_seed": max(0, int(second_seed or 0)),
        "upscale_method": method,
        "upscale_model": upscale_model,
        "has_upscale_model": upscale_model is not None,
        "sample_model": sample_model,
        "has_sample_model": sample_model is not None,
        "latent_upscale_ref": latent_upscale_model,
        "latent_upscale_module": latent_mod,
        "latent_upscale_model": latent_name,
        "h3_latent_model": latent_name,
        "has_latent_upscale_model": latent_upscale_model is not None,
        "sampler": sampler,
        "sigmas": ",".join(f"{x:g}" for x in parsed),
        "sigmas_parsed": parsed,
        "sigmas_tensor": sigma_tensor,
        "has_sigmas_tensor": sigma_tensor is not None,
    }


def normalize_refine_pack(
    raw,
    *,
    base_width: int = 0,
    base_height: int = 0,
) -> dict[str, Any] | None:
    """Director execute: None if unconnected / invalid."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("enabled") is False:
        return None
    mode = str(raw.get("mode") or "refine").strip().lower()
    if mode not in REFINE_MODES:
        mode = "refine"
    ar = normalize_aspect_ratio(raw.get("aspect_ratio"))
    tw, th = int(raw.get("target_width") or 0), int(raw.get("target_height") or 0)
    if mode in {"upscale", "latent_upscale"} and (tw <= 0 or th <= 0):
        if is_scale_by_aspect(ar):
            factor = max(1.0, min(4.0, float(raw.get("scale_by") or 1.5)))
            tw, th = ensure_minimax_canvas(
                max(32, round(base_width * factor)),
                max(32, round(base_height * factor)),
            )
        elif not is_follow_director_aspect(ar) and not is_custom_aspect_ratio(ar):
            resolved = resolution_from_selector(ar, raw.get("megapixels") or DEFAULT_UPSCALE_MEGAPIXELS)
            if resolved:
                tw, th = resolved
        if tw <= 0 or th <= 0:
            tw, th = infer_upscale_target(base_width, base_height)
    elif tw > 0 and th > 0:
        tw, th = ensure_minimax_canvas(tw, th)
    seed_mode = str(raw.get("seed_mode") or "inherit").strip().lower()
    if seed_mode not in SEED_MODES:
        seed_mode = "inherit"
    method = str(raw.get("upscale_method") or "h3_latent").strip().lower()
    if method not in UPSCALE_METHODS:
        method = "h3_latent"
    sampler = str(raw.get("sampler") or DEFAULT_REFINE_SIGMA_SAMPLER).strip() or DEFAULT_REFINE_SIGMA_SAMPLER
    sigma_tensor = raw.get("sigmas_tensor")
    raw_sigmas = raw.get("sigmas")
    if sigma_tensor is None and is_refine_sigmas_tensor(raw_sigmas):
        sigma_tensor = raw_sigmas
    parsed = raw.get("sigmas_parsed") or ()
    if sigma_tensor is not None:
        parsed = parse_refine_sigmas(sigma_tensor, fallback=False)
    elif parsed:
        parsed = parse_refine_sigmas(parsed, fallback=False)
    elif mode != "latent_upscale":
        parsed = generated_refine_sigmas(
            raw.get("second_pass_steps") or 2,
            raw.get("first_sigma") if raw.get("first_sigma") is not None else 0.8,
            raw.get("sigma_spacing") or "linear",
        )
    sample_model = raw.get("sample_model")
    if sample_model is None:
        sample_model = raw.get("model")
    latent_raw = raw.get("latent_upscale_ref")
    if latent_raw is None:
        latent_raw = raw.get("latent_upscale_model")
    latent_mod, latent_name = resolve_latent_upscale_ref(latent_raw)
    if not latent_name:
        latent_name = str(raw.get("h3_latent_model") or "").strip()
    upscale = raw.get("upscale_model")
    device = str(raw.get("latent_upscale_device") or "auto").strip().lower()
    if device not in LATENT_UPSCALE_DEVICES:
        device = "auto"
    precision = str(raw.get("latent_upscale_precision") or "fp16").strip().lower()
    if precision not in LATENT_UPSCALE_PRECISIONS:
        precision = "fp16"
    sigma_spacing = str(raw.get("sigma_spacing") or "linear").strip().lower()
    if sigma_spacing not in SIGMA_SPACINGS:
        sigma_spacing = "linear"
    first_sigma = raw.get("first_sigma")
    if first_sigma is None:
        first_sigma = 0.8
    return {
        "enabled": True,
        "mode": mode,
        "passes": refine_passes_for(raw),
        "seed_mode": seed_mode,
        "aspect_ratio": ar,
        "megapixels": float(raw.get("megapixels") or DEFAULT_UPSCALE_MEGAPIXELS),
        "target_width": tw,
        "target_height": th,
        "skip_fl2v": bool(raw.get("skip_fl2v", True)),
        "scale_by": max(1.0, min(4.0, float(raw.get("scale_by") or 1.5))),
        "latent_upscale_device": device,
        "latent_upscale_precision": precision,
        "strict": bool(raw.get("strict", True)),
        "second_pass_steps": max(1, min(100, int(raw.get("second_pass_steps") or 2))),
        "first_sigma": max(0.0, min(20000.0, float(first_sigma))),
        "sigma_spacing": sigma_spacing,
        "second_seed": max(0, int(raw.get("second_seed") or 0)),
        "upscale_method": method,
        "upscale_model": upscale,
        "has_upscale_model": upscale is not None,
        "sample_model": sample_model,
        "has_sample_model": sample_model is not None,
        "latent_upscale_ref": latent_raw,
        "latent_upscale_module": latent_mod,
        "latent_upscale_model": latent_name,
        "h3_latent_model": latent_name,
        "has_latent_upscale_model": latent_raw is not None,
        "sampler": sampler,
        "sigmas": ",".join(f"{x:g}" for x in parsed),
        "sigmas_parsed": parsed,
        "sigmas_tensor": sigma_tensor,
        "has_sigmas_tensor": sigma_tensor is not None,
    }


def refine_will_sample(plan, seg) -> bool:
    """True when this segment will run refine and/or upscale processing."""
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return False
    return True


def refine_passes_for(pack: dict[str, Any] | None) -> int:
    try:
        n = int((pack or {}).get("passes") or 1)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(MAX_REFINE_PASSES, n))


def refine_model_for(pack: dict[str, Any] | None, fallback):
    """Second-pass UNET. Unconnected Refine.refine_model → Director main model."""
    custom = (pack or {}).get("sample_model")
    if custom is None:
        custom = (pack or {}).get("model")
    return fallback if custom is None else custom


def refine_seed_for(pack: dict[str, Any], seed: int, pass_index: int = 0) -> int:
    if pack.get("seed_mode") == "fixed":
        return int(pack.get("second_seed") or 0)
    if pack.get("seed_mode") == "offset":
        return int(seed) + 1 + int(max(0, pass_index))
    return int(seed)


def refine_fingerprint(plan) -> dict[str, Any]:
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return {"refine": False}
    return {
        "refine": True,
        "refine_mode": pack.get("mode") or "refine",
        "refine_passes": refine_passes_for(pack),
        "refine_seed_mode": pack.get("seed_mode") or "inherit",
        "refine_target": f"{int(pack.get('target_width') or 0)}x{int(pack.get('target_height') or 0)}",
        "refine_aspect": pack.get("aspect_ratio") or FOLLOW_DIRECTOR_ASPECT,
        "refine_megapixels": round(float(pack.get("megapixels") or 0), 3),
        "refine_scale_by": round(float(pack.get("scale_by") or 0), 3),
        "refine_upscale_method": pack.get("upscale_method") or "h3_latent",
        "refine_upscale_model": bool(pack.get("has_upscale_model") or pack.get("upscale_model") is not None),
        "refine_latent_upscale_model": latent_upscale_model_name(pack),
        "refine_sampler": pack.get("sampler") or "",
        "refine_sigmas": ",".join(f"{x:.4f}" for x in (pack.get("sigmas_parsed") or ()))
        if pack.get("has_sigmas_tensor") or pack.get("sigmas_tensor") is not None
        else (pack.get("sigmas") or ""),
        "refine_sigmas_wired": bool(pack.get("has_sigmas_tensor") or pack.get("sigmas_tensor") is not None),
        "refine_upscale_device": pack.get("latent_upscale_device") or "auto",
        "refine_upscale_precision": pack.get("latent_upscale_precision") or "fp16",
        "refine_strict": bool(pack.get("strict", True)),
        "refine_second_seed": int(pack.get("second_seed") or 0),
        "refine_sample_model": bool(pack.get("has_sample_model") or pack.get("sample_model") is not None),
        "refine_skip_fl2v": bool(pack.get("skip_fl2v", True)),
    }


def refine_report_line(plan) -> str | None:
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return None
    mode = pack.get("mode") or "refine"
    extra = ""
    if refine_needs_canvas(pack):
        ar = pack.get("aspect_ratio") or FOLLOW_DIRECTOR_ASPECT
        if refine_uses_h3_latent(pack):
            how = latent_upscale_model_name(pack) or "h3_latent"
        else:
            method = pack.get("upscale_method") or "h3_latent"
            if method == "nvidia_rtx_vsr":
                how = "nvidia_rtx_vsr"
            elif pack.get("has_upscale_model") or pack.get("upscale_model") is not None:
                how = "upscale_model"
            else:
                how = "lanczos"
        extra = (
            f", {ar} → {int(pack.get('target_width') or 0)}×{int(pack.get('target_height') or 0)}"
            f", {how}"
        )
    n_passes = refine_passes_for(pack)
    pass_note = f", passes={n_passes}" if n_passes > 1 else ""
    model_note = (
        ", 二采模型" if (pack.get("has_sample_model") or pack.get("sample_model") is not None) else ""
    )
    wired = bool(pack.get("has_sigmas_tensor") or pack.get("sigmas_tensor") is not None)
    parsed = pack.get("sigmas_parsed") or ()
    sampler = pack.get("sampler") or DEFAULT_REFINE_SIGMA_SAMPLER
    n_steps = max(1, len(parsed) - 1) if parsed else 0
    if mode == "latent_upscale":
        return f"Refine: ON ({mode}{model_note}{extra})"
    how = f"sigmas {sampler}" if wired else f"generated sigmas {sampler}"
    step_note = f" {n_steps}-step" if n_steps else ""
    return (
        f"Refine: ON ({mode}, {how}{step_note}"
        f"{pass_note}{model_note}{extra})"
    )
