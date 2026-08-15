"""MiniMax H3 Director with optional LoRAs and model patches."""

from __future__ import annotations

import json
from typing import Any

import nodes
from comfy_extras.nodes_easycache import EasyCacheNode

from .director import MiniMaxH3Director


SAGE_ATTENTION_MODES = [
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
]


def _first_output(output: Any) -> Any:
    if hasattr(output, "args"):
        values = output.args
    elif isinstance(output, (tuple, list)):
        values = output
    else:
        raise RuntimeError(f"Unexpected node output type: {type(output)!r}")
    if not values:
        raise RuntimeError("Model patch node returned no outputs.")
    return values[0]


def _registered_node(node_id: str) -> type:
    node_class = nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if node_class is None:
        raise RuntimeError(
            f"{node_id} is enabled but its custom node is not loaded. "
            "Install or enable the required custom node and restart ComfyUI."
        )
    return node_class


def _parse_loras(raw: str | list[Any]) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            data = json.loads(raw or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid MiniMax H3 Director LoRA configuration: {exc.msg}.") from exc
    else:
        data = raw
    if not isinstance(data, list):
        raise ValueError("MiniMax H3 Director LoRA configuration must be a list.")

    loras = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"LoRA entry {index} must be an object.")
        if not item.get("enabled", True):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"LoRA entry {index} needs a filename.")
        try:
            strength = float(item.get("strength", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LoRA entry {index} has an invalid strength.") from exc
        if not -100.0 <= strength <= 100.0:
            raise ValueError(f"LoRA entry {index} strength must be between -100 and 100.")
        if strength != 0.0:
            loras.append({"name": name, "strength": strength})
    return loras


def _load_loras(model: Any, raw: str | list[Any]) -> Any:
    for lora in _parse_loras(raw):
        model = nodes.LoraLoaderModelOnly().load_lora_model_only(
            model,
            lora["name"],
            lora["strength"],
        )[0]
    return model


class MiniMaxH3DirectorAdvanced(MiniMaxH3Director):
    """Director variant that prepares its model before running the timeline."""

    DESCRIPTION = (
        "MiniMax H3 Director with ordered model-only LoRAs and optional "
        "attention, Sol-Attn, and EasyCache patches."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        inputs = super().INPUT_TYPES()
        optional = dict(inputs.get("optional", {}))
        optional.update(
            {
                "lora_config": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "LoRA rows managed by the Director model enhancements panel.",
                    },
                ),
                "enable_sage_attention": ("BOOLEAN", {"default": False}),
                "sage_attention": (SAGE_ATTENTION_MODES, {"default": "auto"}),
                "allow_sage_compile": ("BOOLEAN", {"default": False}),
                "enable_h3_mem_eff_sage": ("BOOLEAN", {"default": False}),
                "enable_sol_attn": ("BOOLEAN", {"default": False}),
                "sol_tau": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 4.0, "step": 0.05}),
                "sol_start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sol_end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sol_min_tokens": ("INT", {"default": 4096, "min": 0, "max": 1 << 20, "step": 512}),
                "sol_int8_qk": ("BOOLEAN", {"default": True}),
                "sol_sink_conditioning": (
                    ["exact_kv", "exact_kv_and_rows", "off"],
                    {"default": "exact_kv_and_rows"},
                ),
                "sol_morton": ("BOOLEAN", {"default": False}),
                "sol_morton_curve": (["3d", "2d_frame"], {"default": "2d_frame"}),
                "sol_int8_pv": ("BOOLEAN", {"default": True}),
                "sol_verbose": ("BOOLEAN", {"default": False}),
                "sol_use_tma": ("BOOLEAN", {"default": False}),
                "sol_tau_profile": ("STRING", {"default": "", "multiline": True}),
                "sol_dense_blocks": ("STRING", {"default": ""}),
                "enable_easycache": ("BOOLEAN", {"default": False}),
                "easycache_reuse_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3.0, "step": 0.01}),
                "easycache_start_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "easycache_end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "easycache_verbose": ("BOOLEAN", {"default": False}),
            }
        )
        inputs["optional"] = optional
        return inputs

    def execute(
        self,
        model: Any,
        video_vae: Any,
        audio_vae: Any,
        clip: Any,
        lora_config: str = "[]",
        enable_sage_attention: bool = False,
        sage_attention: str = "auto",
        allow_sage_compile: bool = False,
        enable_h3_mem_eff_sage: bool = False,
        enable_sol_attn: bool = False,
        sol_tau: float = 1.3,
        sol_start_percent: float = 0.2,
        sol_end_percent: float = 0.9,
        sol_min_tokens: int = 4096,
        sol_int8_qk: bool = True,
        sol_sink_conditioning: str = "exact_kv_and_rows",
        sol_morton: bool = False,
        sol_morton_curve: str = "2d_frame",
        sol_int8_pv: bool = True,
        sol_verbose: bool = False,
        sol_use_tma: bool = False,
        sol_tau_profile: str = "",
        sol_dense_blocks: str = "",
        enable_easycache: bool = False,
        easycache_reuse_threshold: float = 0.2,
        easycache_start_percent: float = 0.15,
        easycache_end_percent: float = 0.95,
        easycache_verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        shared_lora_model = _load_loras(model, lora_config)

        def apply_model_patches(candidate: Any) -> Any:
            if enable_sage_attention:
                patcher = _registered_node("PathchSageAttentionKJ")()
                candidate = _first_output(patcher.patch(candidate, sage_attention, allow_sage_compile))

            if enable_h3_mem_eff_sage:
                patcher = _registered_node("MiniMaxH3MemoryEfficientSageAttentionPatch")
                candidate = _first_output(patcher.execute(candidate))

            if enable_sol_attn:
                patcher = _registered_node("SolAttnPatch")
                candidate = _first_output(
                    patcher.execute(
                        candidate,
                        float(sol_tau),
                        float(sol_start_percent),
                        float(sol_end_percent),
                        int(sol_min_tokens),
                        bool(sol_int8_qk),
                        sol_sink_conditioning,
                        bool(sol_morton),
                        sol_morton_curve,
                        sol_dense_blocks,
                        bool(sol_verbose),
                        tau_profile=sol_tau_profile or None,
                        use_tma=bool(sol_use_tma),
                        int8_pv=bool(sol_int8_pv),
                    )
                )

            if enable_easycache:
                candidate = _first_output(
                    EasyCacheNode.execute(
                        candidate,
                        float(easycache_reuse_threshold),
                        float(easycache_start_percent),
                        float(easycache_end_percent),
                        bool(easycache_verbose),
                    )
                )
            return candidate

        shared_model = apply_model_patches(shared_lora_model)

        def segment_model_provider(_model: Any, segment: Any) -> Any:
            local_loras = _parse_loras(getattr(segment, "loras", []) or [])
            if not local_loras:
                return shared_model
            return apply_model_patches(_load_loras(shared_lora_model, local_loras))

        return super().execute(
            model=shared_model,
            video_vae=video_vae,
            audio_vae=audio_vae,
            clip=clip,
            _segment_model_provider=segment_model_provider,
            **kwargs,
        )
