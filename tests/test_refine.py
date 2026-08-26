import math
import re
import types
import unittest
from unittest.mock import patch

import torch

from ComfyUI_MiniMaxH3_Director.nodes.director_refine import MiniMaxH3DirectorRefine
from ComfyUI_MiniMaxH3_Director.director import refine_pack
from ComfyUI_MiniMaxH3_Director.director import refine_sampling


class FakeNodeOutput:
    def __init__(self, value):
        self.args = (value,)


class RefinePackTest(unittest.TestCase):
    def test_refine_tooltips_are_english(self):
        input_types = MiniMaxH3DirectorRefine.INPUT_TYPES()
        fields = [*input_types["required"].values(), *input_types["optional"].values()]
        tooltips = [field[1]["tooltip"] for field in fields]

        self.assertEqual(len(tooltips), 35)
        self.assertTrue(all(tooltips))
        self.assertFalse(any(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", tooltip) for tooltip in tooltips))

    def test_aspect_choices_use_english_labels(self):
        self.assertEqual(
            refine_pack.ASPECT_RATIO_CHOICES,
            (
                "Follow Director",
                "Scale by multiplier",
                "1:1 (Square)",
                "2:3 (Portrait photo)",
                "3:2 (Landscape photo)",
                "3:4 (Portrait standard)",
                "4:3 (Standard)",
                "9:16 (Portrait)",
                "16:9 (Widescreen)",
                "21:9 (Ultrawide)",
                "Custom",
            ),
        )

    def test_chinese_aspect_values_normalize_to_english(self):
        legacy = {
            "跟随导演台": "Follow Director",
            "按倍数": "Scale by multiplier",
            "1:1 (方形)": "1:1 (Square)",
            "2:3 (竖版照片)": "2:3 (Portrait photo)",
            "3:2 (横版照片)": "3:2 (Landscape photo)",
            "3:4 (竖版标准)": "3:4 (Portrait standard)",
            "4:3 (标准)": "4:3 (Standard)",
            "9:16 (竖屏)": "9:16 (Portrait)",
            "16:9 (宽屏)": "16:9 (Widescreen)",
            "21:9 (超宽)": "21:9 (Ultrawide)",
            "自定义": "Custom",
        }
        for old, expected in legacy.items():
            with self.subTest(old=old):
                self.assertEqual(refine_pack.normalize_aspect_ratio(old), expected)

    def test_legacy_and_english_ratio_values_resolve_identically(self):
        self.assertEqual(
            refine_pack.resolution_from_selector("16:9 (宽屏)", 2.0),
            refine_pack.resolution_from_selector("16:9 (Widescreen)", 2.0),
        )

    def test_scale_by_resolves_reference_workflow_canvas(self):
        raw = refine_pack.pack_refine(
            mode="upscale",
            aspect_ratio=refine_pack.SCALE_BY_ASPECT,
            scale_by=1.6,
            latent_upscale_model="minimax_h3_latent_upscaler_3d_fp16.safetensors",
        )

        normalized = refine_pack.normalize_refine_pack(
            raw,
            base_width=480,
            base_height=800,
        )

        self.assertEqual((normalized["target_width"], normalized["target_height"]), (768, 1280))

    def test_generated_sigmas_match_extend_intermediate_sigmas(self):
        self.assertEqual(
            refine_pack.generated_refine_sigmas(2, 0.8, "linear"),
            (0.8, 0.4, 0.0),
        )
        cosine = refine_pack.generated_refine_sigmas(2, 0.8, "cosine")
        self.assertAlmostEqual(cosine[1], 0.8 * (1 - math.sin(math.pi / 4)))

    def test_wired_sigmas_override_generated_schedule(self):
        wired = torch.tensor([0.7, 0.2, 0.0])
        pack = refine_pack.pack_refine(mode="refine", sigmas=wired, first_sigma=0.8)

        self.assertEqual(
            len(refine_pack.refine_sigmas_override(pack)),
            3,
        )
        for actual, expected in zip(
            refine_pack.refine_sigmas_override(pack),
            (0.7, 0.2, 0.0),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

    def test_fixed_second_pass_seed_is_independent(self):
        pack = {"seed_mode": "fixed", "second_seed": 42}

        self.assertEqual(refine_pack.refine_seed_for(pack, 999, pass_index=3), 42)

    def test_fl2v_still_enters_refine_for_final_resolution(self):
        plan = types.SimpleNamespace(refine={"enabled": True, "skip_fl2v": True})
        segment = types.SimpleNamespace(task_key="fl2v")

        self.assertTrue(refine_pack.refine_will_sample(plan, segment))

    def test_tiled_refine_is_disabled_by_default(self):
        pack = refine_pack.pack_refine()

        self.assertFalse(pack["tiled_refine_enabled"])
        self.assertTrue(pack["tiled_refine_temporal"])
        self.assertTrue(pack["tiled_refine_spatial"])
        self.assertEqual(pack["tiled_refine_chunk_length"], 85)
        self.assertEqual(pack["tiled_refine_tile_width"], 480)
        self.assertEqual(pack["tiled_refine_tile_height"], 864)

    def test_tiled_refine_settings_survive_normalization(self):
        pack = refine_pack.pack_refine(
            tiled_refine_enabled=True,
            tiled_refine_chunk_length=102,
            tiled_refine_temporal_overlap=34,
            tiled_refine_tile_width=640,
            tiled_refine_tile_height=960,
            tiled_refine_overlap_mode="later",
            tiled_refine_blend="smoothstep",
        )

        normalized = refine_pack.normalize_refine_pack(pack)

        self.assertTrue(normalized["tiled_refine_enabled"])
        self.assertEqual(normalized["tiled_refine_chunk_length"], 102)
        self.assertEqual(normalized["tiled_refine_temporal_overlap"], 34)
        self.assertEqual(normalized["tiled_refine_tile_width"], 640)
        self.assertEqual(normalized["tiled_refine_tile_height"], 960)
        self.assertEqual(normalized["tiled_refine_overlap_mode"], "later")
        self.assertEqual(normalized["tiled_refine_blend"], "smoothstep")

    def test_tiled_refine_zero_overlaps_are_preserved(self):
        pack = refine_pack.pack_refine(
            tiled_refine_temporal_overlap=0,
            tiled_refine_tile_overlap=0,
            tiled_refine_fade=0,
        )

        self.assertEqual(pack["tiled_refine_temporal_overlap"], 0)
        self.assertEqual(pack["tiled_refine_tile_overlap"], 0)
        self.assertEqual(pack["tiled_refine_fade"], 0)

    def test_tiled_refine_settings_change_cache_fingerprint_and_report(self):
        disabled = types.SimpleNamespace(refine=refine_pack.pack_refine())
        enabled = types.SimpleNamespace(
            refine=refine_pack.pack_refine(tiled_refine_enabled=True)
        )

        disabled_fingerprint = refine_pack.refine_fingerprint(disabled)
        enabled_fingerprint = refine_pack.refine_fingerprint(enabled)

        self.assertFalse(disabled_fingerprint["refine_tiled"])
        self.assertTrue(enabled_fingerprint["refine_tiled"])
        self.assertNotEqual(disabled_fingerprint, enabled_fingerprint)
        self.assertIn("tiled 85f/17f+480×864/128", refine_pack.refine_report_line(enabled))


class RefineSamplingTest(unittest.TestCase):
    def test_external_upscaler_receives_resolved_target_and_compute_options(self):
        calls = []

        class ExternalUpscaler:
            @classmethod
            def execute(
                cls,
                latent,
                model_name,
                mode,
                align,
                enable_chunking,
                device,
                precision,
            ):
                calls.append(
                    (latent, model_name, mode, align, enable_chunking, device, precision)
                )
                return FakeNodeOutput({"samples": torch.zeros((1, 24, 2, 4, 8))})

        fake_nodes = types.SimpleNamespace(
            NODE_CLASS_MAPPINGS={"MinimaxH3LatentUpscaler3D": ExternalUpscaler}
        )
        video = {"samples": torch.zeros((1, 24, 2, 2, 4))}
        audio = {"samples": torch.ones((1, 8))}
        plan = types.SimpleNamespace(width=64, height=32)
        pack = {
            "latent_upscale_model": "upscaler.safetensors",
            "latent_upscale_device": "cpu",
            "latent_upscale_precision": "fp16",
            "strict": True,
        }

        with patch.dict("sys.modules", {"nodes": fake_nodes}), patch.object(
            refine_sampling, "_split_av", return_value=(video, audio)
        ), patch.object(
            refine_sampling,
            "_join_av",
            side_effect=lambda encoded, audio_latent, _template: {
                "samples": encoded["samples"],
                "audio": audio_latent,
            },
        ):
            result, _, notes = refine_sampling._apply_h3_latent_upscale(
                {"samples": object()},
                pack,
                plan=plan,
                tw=128,
                th=64,
                first_pass_images=None,
                pin_frames=0,
                task_key="t2v",
                vae=None,
                refine_positive=[],
            )

        self.assertIs(result["audio"]["samples"], audio["samples"])
        self.assertEqual(notes, ["128×64", "h3_latent"])
        self.assertIs(calls[0][0], video)
        self.assertEqual(
            calls[0][1:],
            (
                "upscaler.safetensors",
                {"mode": "target dimensions", "width": 128, "height": 64},
                32,
                True,
                "cpu",
                "fp16",
            ),
        )

    def test_strict_refine_raises_instead_of_returning_first_pass(self):
        samples = {"samples": torch.zeros((1, 24, 2, 2, 4))}
        plan = types.SimpleNamespace(
            refine={
                "enabled": True,
                "mode": "upscale",
                "upscale_method": "h3_latent",
                "strict": True,
                "target_width": 128,
                "target_height": 64,
            }
        )
        segment = types.SimpleNamespace(index=0, task_key="t2v")

        with patch.object(
            refine_sampling,
            "_apply_h3_latent_upscale",
            side_effect=ValueError("backend unavailable"),
        ), self.assertRaisesRegex(RuntimeError, "Segment 1 refine failed"):
            refine_sampling.apply_segment_refine(
                plan,
                segment,
                samples=samples,
                model=object(),
                vae=object(),
                positive=[],
                negative=[],
                seed=0,
                cfg=1.0,
                first_steps=4,
                sampler_name="euler",
                scheduler="simple",
                shift_video=12.0,
                shift_audio=3.0,
            )

    def test_enabled_tiled_refine_replaces_full_second_pass(self):
        samples = {"samples": object()}
        refined = {"samples": object()}
        plan = types.SimpleNamespace(
            refine={
                "enabled": True,
                "mode": "refine",
                "passes": 1,
                "sampler": "euler",
                "sigmas_parsed": (0.8, 0.4, 0.0),
                "tiled_refine_enabled": True,
                "tiled_refine_temporal": True,
                "tiled_refine_spatial": True,
            }
        )
        segment = types.SimpleNamespace(index=0, task_key="t2v")

        with patch.object(
            refine_sampling,
            "sample_tiled_refine",
            return_value=refined,
        ) as tiled, patch.object(refine_sampling, "sample_single_stage") as full:
            result, note = refine_sampling.apply_segment_refine(
                plan,
                segment,
                samples=samples,
                model=object(),
                vae=object(),
                positive=[],
                negative=[],
                seed=0,
                cfg=1.0,
                first_steps=4,
                sampler_name="euler",
                scheduler="simple",
                shift_video=12.0,
                shift_audio=3.0,
            )

        self.assertIs(result, refined)
        self.assertIn("tiled=85f/0f+480×864", note)
        tiled.assert_called_once()
        full.assert_not_called()


if __name__ == "__main__":
    unittest.main()
