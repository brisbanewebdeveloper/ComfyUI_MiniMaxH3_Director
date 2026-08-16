"""First-block residual caching for ComfyUI's MiniMax H3 DiT."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Callable

import torch

import comfy.model_patcher
import comfy.patcher_extension


_CACHE_KEY = "minimax_h3_firstblock_cache"
_WRAPPER_KEY = "minimax_h3_firstblock_cache"
_MAX_RETAINED_CACHE_BYTES = 768 * 1024**2
_MIN_DEVICE_HEADROOM_BYTES = 4 * 1024**3
_DEVICE_HEADROOM_FRACTION = 0.15
_log = logging.getLogger("ComfyUI-MiniMaxH3-Director.firstblock_cache")


@dataclass
class _CacheState:
    previous_head_residual: torch.Tensor | None = None
    tail_residual: torch.Tensor | None = None
    pending_head_output: torch.Tensor | None = None
    skip_tail: bool = False
    cache_disabled: bool = False


def _execution_key(transformer_options: dict[str, Any]) -> tuple[str, ...]:
    uuids = transformer_options.get("uuids")
    if uuids:
        return ("uuid", *(str(value) for value in uuids))
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    if cond_or_uncond:
        return ("cond", *(str(value) for value in cond_or_uncond))
    return ("default",)


def _relative_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    totals = torch.stack(
        (
            (current - previous).abs().sum(dtype=torch.float32),
            previous.abs().sum(dtype=torch.float32),
        )
    )
    return float((totals[0] / totals[1].clamp_min(1.0e-8)).item())


def _cache_bypass_reason(hidden: torch.Tensor) -> str | None:
    tensor_bytes = hidden.numel() * hidden.element_size()
    retained_bytes = tensor_bytes * 2
    if retained_bytes > _MAX_RETAINED_CACHE_BYTES:
        return (
            f"estimated retained cache {retained_bytes / 1024**2:.0f} MiB exceeds "
            f"the conservative {_MAX_RETAINED_CACHE_BYTES / 1024**2:.0f} MiB limit"
        )

    if hidden.device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(hidden.device)
        reserve_bytes = max(_MIN_DEVICE_HEADROOM_BYTES, int(total_bytes * _DEVICE_HEADROOM_FRACTION))
        # Initial population also needs a saved block input and a fresh residual
        # before the two retained tensors settle into their steady state.
        working_bytes = tensor_bytes * 3
        if free_bytes - working_bytes < reserve_bytes:
            return (
                f"estimated retained cache {retained_bytes / 1024**2:.0f} MiB would leave "
                f"less than {reserve_bytes / 1024**3:.1f} GiB of device headroom"
            )

    return None


class FirstBlockCacheHolder:
    """Own per-sampling residual state and observable cache statistics."""

    def __init__(self, threshold: float, verbose: bool = False) -> None:
        if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("FirstBlockCache threshold must be greater than zero and at most one")
        self.threshold = threshold
        self.verbose = verbose
        self.states: dict[tuple[str, ...], _CacheState] = {}
        self.calls = 0
        self.compute = 0
        self.reuse = 0

    def clone(self) -> FirstBlockCacheHolder:
        return FirstBlockCacheHolder(self.threshold, self.verbose)

    def reset(self) -> None:
        self.states.clear()
        self.calls = 0
        self.compute = 0
        self.reuse = 0

    def state_for(self, transformer_options: dict[str, Any]) -> _CacheState:
        return self.states.setdefault(_execution_key(transformer_options), _CacheState())

    def prepare_head(
        self,
        head_input: torch.Tensor,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor | None:
        state = self.state_for(transformer_options)
        if state.cache_disabled:
            return None

        cached = state.previous_head_residual
        if cached is not None and (
            cached.shape != head_input.shape
            or cached.dtype != head_input.dtype
            or cached.device != head_input.device
        ):
            state.previous_head_residual = None
            state.tail_residual = None

        if state.previous_head_residual is None and state.tail_residual is None:
            bypass_reason = _cache_bypass_reason(head_input)
            if bypass_reason is not None:
                state.cache_disabled = True
                _log.warning("FirstBlockCache bypassed for this execution: %s.", bypass_reason)
                return None

        # MiniMax H3 blocks update their input in place, so preserve the value
        # used to calculate the first-block residual before running the block.
        return head_input.detach().clone()

    def after_head(
        self,
        head_input: torch.Tensor | None,
        head_output: torch.Tensor,
        transformer_options: dict[str, Any],
    ) -> tuple[torch.Tensor, bool]:
        state = self.state_for(transformer_options)
        state.pending_head_output = None
        state.skip_tail = False
        self.calls += 1

        if state.cache_disabled or head_input is None:
            state.cache_disabled = True
            self.compute += 1
            return head_output, False

        head_residual = (head_output - head_input).detach()

        cached = state.previous_head_residual
        if cached is not None and (
            cached.shape != head_residual.shape
            or cached.dtype != head_residual.dtype
            or cached.device != head_residual.device
        ):
            state.previous_head_residual = None
            state.tail_residual = None
            cached = None

        relative_l1 = None
        if cached is not None and state.tail_residual is not None:
            relative_l1 = _relative_l1(head_residual, cached)
            if relative_l1 <= self.threshold:
                state.skip_tail = True
                self.reuse += 1
                if self.verbose:
                    _log.info(
                        "FirstBlockCache reusing tail: relative_l1=%.6f threshold=%.6f",
                        relative_l1,
                        self.threshold,
                    )
                return head_output + state.tail_residual, True

        self.compute += 1
        # A refresh no longer needs the previous tail. Releasing it here avoids
        # retaining three full hidden-state tensors while the block stack runs.
        state.tail_residual = None
        state.previous_head_residual = head_residual
        state.pending_head_output = head_output.detach().clone()
        if self.verbose:
            detail = "initial refresh" if relative_l1 is None else f"relative_l1={relative_l1:.6f}"
            _log.info("FirstBlockCache computing tail: %s threshold=%.6f", detail, self.threshold)
        return head_output, False

    def after_tail(self, hidden: torch.Tensor, transformer_options: dict[str, Any]) -> torch.Tensor:
        state = self.state_for(transformer_options)
        if state.cache_disabled:
            return hidden
        if state.pending_head_output is None:
            raise RuntimeError("FirstBlockCache completed a refresh without a first-block output")
        state.tail_residual = (hidden - state.pending_head_output).detach()
        state.pending_head_output = None
        return hidden

    def log_summary(self) -> None:
        reuse_rate = self.reuse / self.calls if self.calls else 0.0
        _log.info(
            "FirstBlockCache - reused %d/%d block-stack calls (%.1f%%), threshold %.4f.",
            self.reuse,
            self.calls,
            reuse_rate * 100.0,
            self.threshold,
        )


def _sample_wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
    guider = executor.class_obj
    original_options = guider.model_options
    run_options = comfy.model_patcher.create_model_options_clone(original_options)
    holder = run_options["transformer_options"][_CACHE_KEY].clone()
    run_options["transformer_options"][_CACHE_KEY] = holder
    guider.model_options = run_options
    _log.info("FirstBlockCache enabled - threshold: %.4f", holder.threshold)
    try:
        return executor(*args, **kwargs)
    finally:
        holder.log_summary()
        holder.reset()
        guider.model_options = original_options


class _BlockPatch:
    def __init__(
        self,
        index: int,
        last_index: int,
        previous: Callable[[dict[str, Any], dict[str, Any]], dict[str, torch.Tensor]] | None,
    ) -> None:
        self.index = index
        self.last_index = last_index
        self.previous = previous

    def _run_block(self, args: dict[str, Any], extra_options: dict[str, Any]) -> torch.Tensor:
        result = (
            self.previous(args, extra_options)
            if self.previous is not None
            else extra_options["original_block"](args)
        )
        return result["img"]

    def __call__(self, args: dict[str, Any], extra_options: dict[str, Any]) -> dict[str, torch.Tensor]:
        transformer_options = args["transformer_options"]
        holder: FirstBlockCacheHolder = transformer_options[_CACHE_KEY]

        if self.index == 0:
            head_input = holder.prepare_head(args["img"], transformer_options)
            head_output = self._run_block(args, extra_options)
            hidden, _ = holder.after_head(head_input, head_output, transformer_options)
            return {"img": hidden}

        state = holder.state_for(transformer_options)
        if state.skip_tail:
            return {"img": args["img"]}

        hidden = self._run_block(args, extra_options)
        if self.index == self.last_index:
            hidden = holder.after_tail(hidden, transformer_options)
        return {"img": hidden}


def apply_first_block_cache(model: Any, threshold: float = 0.08, verbose: bool = False) -> Any:
    """Clone and patch a MiniMax H3 model with FirstBlockCache."""
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion_model is None or diffusion_model.__class__.__name__ != "MiniMaxH3Model":
        raise ValueError("FirstBlockCache requires a MiniMax H3 diffusion model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) < 2:
        raise ValueError("FirstBlockCache requires the MiniMax H3 DiT block stack")

    patched = model.clone()
    transformer_options = patched.model_options["transformer_options"]
    if "easycache" in transformer_options:
        raise ValueError("FirstBlockCache cannot be combined with EasyCache; enable only one cache")
    if _CACHE_KEY in transformer_options:
        raise ValueError("FirstBlockCache is already enabled on this model")

    existing = dict(transformer_options.get("patches_replace", {}).get("dit", {}))
    transformer_options[_CACHE_KEY] = FirstBlockCacheHolder(float(threshold), bool(verbose))
    last_index = len(blocks) - 1
    for index in range(len(blocks)):
        previous = existing.get(("double_block", index))
        patched.set_model_patch_replace(
            _BlockPatch(index, last_index, previous),
            "dit",
            "double_block",
            index,
        )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        _WRAPPER_KEY,
        _sample_wrapper,
    )
    return patched


__all__ = ["FirstBlockCacheHolder", "apply_first_block_cache"]
