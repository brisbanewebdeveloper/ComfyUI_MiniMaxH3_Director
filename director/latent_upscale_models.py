"""Model discovery for the external MiniMax H3 latent-upscaler node."""

from __future__ import annotations

from pathlib import Path

import folder_paths

LATENT_UPSCALE_FOLDER = "latent_upscale_models"
MISSING_MODEL_LABEL = "(将 3D 权重放入 models/latent_upscale_models)"


def ensure_latent_upscale_folder() -> None:
    """Register the shared model folder used by the external upscaler."""
    if LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            LATENT_UPSCALE_FOLDER,
            str(Path(folder_paths.models_dir) / LATENT_UPSCALE_FOLDER),
        )


def list_h3_latent_upscale_models() -> list[str]:
    """Return available 3D checkpoint filenames for the Refine combo."""
    try:
        ensure_latent_upscale_folder()
        names = folder_paths.get_filename_list(LATENT_UPSCALE_FOLDER) or []
    except (AttributeError, OSError):
        names = []
    supported = sorted(
        name
        for name in names
        if "3d" in name.lower() and Path(name).suffix.lower() in {".safetensors", ".pth"}
    )
    return supported or [MISSING_MODEL_LABEL]
