"""Graph packer: Refine / upscale config for MiniMax H3 Director.refine."""

from __future__ import annotations

import comfy.samplers

from ..director.latent_upscale_models import list_h3_latent_upscale_models
from ..director.refine_pack import (
    ASPECT_RATIO_CHOICES,
    DEFAULT_REFINE_SIGMA_SAMPLER,
    DEFAULT_UPSCALE_MEGAPIXELS,
    FOLLOW_DIRECTOR_ASPECT,
    LATENT_UPSCALE_DEVICES,
    LATENT_UPSCALE_PRECISIONS,
    MAX_REFINE_PASSES,
    MMX_DIR_REFINE,
    REFINE_MODES,
    SEED_MODES,
    SIGMA_SPACINGS,
    UPSCALE_METHODS,
    infer_upscale_target,
    pack_refine,
)

_CATEGORY = "MiniMaxH3"


class MiniMaxH3DirectorRefine:
    """Pack refine/upscale settings. Connect ``refine`` to Director.refine.

    ``refine``: same-resolution second sample.
    ``upscale``: enlarge to target canvas then second-sample.
    ``latent_upscale``: H3 latent enlarge only, no second sample.
    Second sample uses SIGMAS from BasicScheduler / ManualSigmas.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    list(REFINE_MODES),
                    {
                        "default": "refine",
                        "tooltip": (
                            "refine = same-resolution second pass (detail enhancement). "
                            "upscale = upscale to the target canvas, then run a second pass. "
                            "latent_upscale = upscale the H3 latent only; no second pass."
                        ),
                    },
                ),
                "upscale_method": (
                    list(UPSCALE_METHODS),
                    {
                        "default": "h3_latent",
                        "tooltip": (
                            "Used only when mode=upscale. "
                            "h3_latent = upscale the H3 video latent to the target canvas, then run a second pass "
                            "using the 3D weights below. "
                            "lanczos = pixel interpolation; an upscale_model such as RealESRGAN can be connected. "
                            "nvidia_rtx_vsr = NVIDIA RTX Video Super Resolution (requires nvidia-vfx and an NVIDIA GPU)."
                        ),
                    },
                ),
                "latent_upscale_model": (
                    list_h3_latent_upscale_models(),
                    {
                        "tooltip": (
                            "H3 3D latent upscaler weights. "
                            "Enable Comfyui_Minimax_h3_latent_Upscaler and place the weights in "
                            "ComfyUI/models/latent_upscale_models/. "
                            "The filename must contain 3d, for example "
                            "minimax_h3_latent_upscaler_3d_*.safetensors. "
                            "Used with mode=latent_upscale or upscale + h3_latent."
                        ),
                    },
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": DEFAULT_REFINE_SIGMA_SAMPLER,
                        "tooltip": (
                            "Second-pass sampler. The Hailuo example uses euler; "
                            "res_multistep is commonly used for high-quality BasicScheduler second passes."
                        ),
                    },
                ),
                "passes": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_REFINE_PASSES,
                        "tooltip": (
                            "Number of refinement passes. 1 = one second pass. "
                            "With upscale, only the first pass upscales; later passes refine at the same resolution. "
                            "latent_upscale does not run a second pass, so this value has no effect."
                        ),
                    },
                ),
            },
            "optional": {
                "refine_model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Second-pass UNET. If unconnected, the Director's main model is used. "
                            "Useful for applying a Turbo LoRA on the first pass and removing it or switching models on the second."
                        ),
                    },
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Optional second-pass noise schedule. When connected to ComfyUI BasicScheduler, ManualSigmas, "
                            "or custom SIGMAS, it takes priority over the built-in second-pass settings below. "
                            "Connect BasicScheduler to the same MODEL used for the second pass (the Director's main model or refine_model). "
                            "Refine still applies the H3 SigmaShift internally."
                        ),
                    },
                ),
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {
                        "tooltip": (
                            "Optional. Connect an upscaler from Load Upscale Model, such as RealESRGAN_x2plus. "
                            "Used only with mode=upscale and upscale_method=lanczos. "
                            "If unconnected, pure lanczos interpolation is used. Ignored with nvidia_rtx_vsr or h3_latent."
                        ),
                    },
                ),
                "seed_mode": (
                    list(SEED_MODES),
                    {
                        "default": "inherit",
                        "tooltip": "inherit = use the Director seed; offset = add 1, 2, and so on for each pass.",
                    },
                ),
                "aspect_ratio": (
                    list(ASPECT_RATIO_CHOICES),
                    {
                        "default": FOLLOW_DIRECTOR_ASPECT,
                        "tooltip": (
                            "Target canvas for upscaling, using the same algorithm as the Director's output resolution. "
                            "The Director resolution is the first-pass size (for example, 0.4 MP); this is the enlarged target "
                            "(for example, 1.0 MP). Follow Director derives a 720p-class target from the Director canvas ratio. "
                            "Ratio presets work with megapixels. Custom uses the width and height fields, aligned to ×32."
                        ),
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": DEFAULT_UPSCALE_MEGAPIXELS,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": (
                            "Megapixels, matching the Director ResolutionSelector. "
                            "1.0 MP at 16:9 is approximately 1376×768 (aligned to 32). Used only with ratio presets."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 0,
                        "max": 8192,
                        "step": 32,
                        "tooltip": "Custom width (×32). Used only with Custom aspect ratio.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 720,
                        "min": 0,
                        "max": 8192,
                        "step": 32,
                        "tooltip": "Custom height (×32). Used only with Custom aspect ratio.",
                    },
                ),
                "skip_fl2v": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "For first-and-last-frame (fl2v) clips, upscale to the target resolution but skip the second pass "
                            "to protect the keyframes. Disabled by default: rebuild keyframe conditioning after upscaling and run the second pass."
                        ),
                    },
                ),
                "scale_by": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Used when aspect_ratio=Scale by multiplier; the final width and height are still aligned to ×32.",
                    },
                ),
                "latent_upscale_device": (
                    list(LATENT_UPSCALE_DEVICES),
                    {
                        "default": "auto",
                        "tooltip": "H3 3D latent upscaler device. auto prefers CUDA and otherwise uses the CPU.",
                    },
                ),
                "latent_upscale_precision": (
                    list(LATENT_UPSCALE_PRECISIONS),
                    {
                        "default": "fp16",
                        "tooltip": "H3 3D latent upscaler compute precision. fp16 is the default speed/VRAM balance.",
                    },
                ),
                "strict": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Stop the workflow if upscaling or the second pass fails, avoiding a silent low-resolution first-pass result.",
                    },
                ),
                "second_pass_steps": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Built-in second-pass steps when SIGMAS is unconnected. multiple_steps_test uses 2.",
                    },
                ),
                "first_sigma": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 20000.0,
                        "step": 0.01,
                        "tooltip": "First sigma when SIGMAS is unconnected (the workflow denoise value).",
                    },
                ),
                "sigma_spacing": (
                    list(SIGMA_SPACINGS),
                    {
                        "default": "linear",
                        "tooltip": "Distribution of intermediate sigmas when SIGMAS is unconnected.",
                    },
                ),
                "second_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Second-pass seed when seed_mode=fixed.",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **_kwargs):
        # Skip combo/min checks so old workflows (target_width=0 → aspect_ratio) can load.
        return True

    RETURN_TYPES = (MMX_DIR_REFINE, "INT", "INT")
    RETURN_NAMES = ("refine", "width", "height")
    FUNCTION = "pack"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "MiniMax H3 Director Refine: connect to Director.refine. "
        "Director.images is the refined / upscaled result; "
        "Director.images_pre_refine is the first-pass video (before second sample). "
        "Second sample uses SIGMAS from BasicScheduler / ManualSigmas. "
        "When SIGMAS is unconnected, built-in steps / first sigma / spacing are used. "
        "Upscale / latent_upscale canvas uses the same aspect + megapixels / custom W×H as Director. "
        "Director first-pass stays at its own resolution; Refine target is the enlarge size. "
        "width / height are the resolved target canvas (×32). "
        "Does not sample by itself — no IMAGE output."
    )

    def pack(
        self,
        mode="refine",
        upscale_method="h3_latent",
        sampler="",
        passes=1,
        seed_mode="inherit",
        aspect_ratio=FOLLOW_DIRECTOR_ASPECT,
        megapixels=DEFAULT_UPSCALE_MEGAPIXELS,
        width=1280,
        height=720,
        skip_fl2v=False,
        scale_by=1.5,
        latent_upscale_device="auto",
        latent_upscale_precision="fp16",
        strict=True,
        second_pass_steps=2,
        first_sigma=0.8,
        sigma_spacing="linear",
        second_seed=0,
        latent_upscale_model=None,
        upscale_model=None,
        h3_latent_model="",
        sigmas=None,
        refine_model=None,
        model=None,
        target_width=0,
        target_height=0,
        **kwargs,
    ):
        del kwargs
        try:
            mp = float(megapixels)
        except (TypeError, ValueError):
            mp = DEFAULT_UPSCALE_MEGAPIXELS
        if mp < 0.1:
            mp = DEFAULT_UPSCALE_MEGAPIXELS
        try:
            w = int(width or 0)
        except (TypeError, ValueError):
            w = 1280
        try:
            h = int(height or 0)
        except (TypeError, ValueError):
            h = 720
        if w < 32:
            w = 1280
        if h < 32:
            h = 720
        try:
            n_passes = int(passes or 1)
        except (TypeError, ValueError):
            n_passes = 1
        if n_passes < 1:
            n_passes = 1
        pack = pack_refine(
            mode=mode,
            passes=n_passes,
            seed_mode=seed_mode,
            aspect_ratio=aspect_ratio,
            megapixels=mp,
            width=w,
            height=h,
            target_width=target_width,
            target_height=target_height,
            skip_fl2v=skip_fl2v,
            scale_by=scale_by,
            latent_upscale_device=latent_upscale_device,
            latent_upscale_precision=latent_upscale_precision,
            strict=strict,
            second_pass_steps=second_pass_steps,
            first_sigma=first_sigma,
            sigma_spacing=sigma_spacing,
            second_seed=second_seed,
            upscale_method=upscale_method,
            sample_model=refine_model if refine_model is not None else model,
            latent_upscale_model=latent_upscale_model if latent_upscale_model is not None else h3_latent_model,
            upscale_model=upscale_model,
            sampler=sampler,
            sigmas=sigmas,
        )
        out_w = int(pack.get("target_width") or 0)
        out_h = int(pack.get("target_height") or 0)
        if out_w <= 0 or out_h <= 0:
            out_w, out_h = infer_upscale_target(0, 0)
        return (pack, int(out_w), int(out_h))
