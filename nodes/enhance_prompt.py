"""Standalone prompt enhancement node backed by the Director's existing client."""

from __future__ import annotations

from ..director.prompt_enhance_media import sample_video_tensor_frames
from ..lib.prompt_enhance_templates import OUTPUT_LANGUAGE_EN, OUTPUT_LANGUAGE_ZH
from ..lib.prompt_enhancer import (
    API_FORMAT_OLLAMA,
    API_FORMAT_OPENAI_COMPAT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OPENAI_COMPAT_MODE_LLAMA_SWAP,
    OPENAI_COMPAT_MODE_STANDARD,
    enhance_prompt_sync,
)
from ..lib.task_prompts import task_type_combo_options


class MiniMaxH3DirectorEnhancePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        task_options, task_meta = task_type_combo_options()
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "task_type": (task_options, task_meta),
                "api_format": (
                    [API_FORMAT_OLLAMA, API_FORMAT_OPENAI_COMPAT],
                    {"default": API_FORMAT_OLLAMA},
                ),
                "url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "model": ("STRING", {"default": DEFAULT_OLLAMA_MODEL}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Optional vision references; up to four frames are sent."}),
                "openai_compat_mode": (
                    [OPENAI_COMPAT_MODE_STANDARD, OPENAI_COMPAT_MODE_LLAMA_SWAP],
                    {"default": OPENAI_COMPAT_MODE_STANDARD},
                ),
                "output_language": (
                    [OUTPUT_LANGUAGE_EN, OUTPUT_LANGUAGE_ZH],
                    {"default": OUTPUT_LANGUAGE_EN},
                ),
                "character_feature_enhance": ("BOOLEAN", {"default": False}),
                "custom_template": ("STRING", {"default": "", "multiline": True}),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "error")
    FUNCTION = "enhance"
    CATEGORY = "MiniMaxH3"
    DESCRIPTION = "Enhance a MiniMax H3 prompt with local Ollama or an unauthenticated OpenAI-compatible endpoint."

    def enhance(
        self,
        prompt,
        task_type,
        api_format,
        url,
        model,
        images=None,
        openai_compat_mode=OPENAI_COMPAT_MODE_STANDARD,
        output_language=OUTPUT_LANGUAGE_EN,
        character_feature_enhance=False,
        custom_template="",
        unload_after=False,
    ):
        images_b64 = sample_video_tensor_frames(images, max_frames=4) if images is not None else None
        enhanced, error = enhance_prompt_sync(
            task_type=task_type,
            user_prompt=prompt,
            url=url,
            model=model,
            api_format=api_format,
            openai_compat_mode=openai_compat_mode,
            images_b64=images_b64,
            image_num=len(images_b64 or []),
            custom_template=custom_template,
            output_language=output_language,
            character_feature_enhance=character_feature_enhance,
            ref_slots=list(range(len(images_b64 or []))),
            unload_after=unload_after,
        )
        return (enhanced or prompt, error or "")
