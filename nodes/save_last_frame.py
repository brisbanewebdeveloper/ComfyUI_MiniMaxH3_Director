"""Output node that saves only the final frame of an image batch."""

from __future__ import annotations

import nodes as comfy_nodes


class MiniMaxH3DirectorSaveLastFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "MiniMaxH3/last_frame"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_frame",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMaxH3"
    DESCRIPTION = "Save the final frame of a generated video through ComfyUI's standard PNG output path."

    def save(self, images, filename_prefix="MiniMaxH3/last_frame", prompt=None, extra_pnginfo=None):
        if images is None or images.shape[0] <= 0:
            raise ValueError("MiniMax H3 Save Last Frame requires at least one image.")
        return comfy_nodes.SaveImage().save_images(
            images[-1:],
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
