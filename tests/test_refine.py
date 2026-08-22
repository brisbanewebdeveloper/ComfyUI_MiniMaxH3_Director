import math
import types
import unittest
from unittest.mock import patch

import torch

from ComfyUI_MiniMaxH3_Director.director import refine_pack
from ComfyUI_MiniMaxH3_Director.director import refine_sampling


class FakeNodeOutput:
    def __init__(self, value):
        self.args = (value,)


class RefinePackTest(unittest.TestCase):
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


class RefineSamplingTest(unittest.TestCase):
    def test_external_upscaler_receives_resolved_target_and_compute_options(self):
        calls = []

        class ExternalUpscaler:
            @classmethod
            def execute(cls, latent, model_name, mode, align, device, precision):
                calls.append((latent, model_name, mode, align, device, precision))
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
        self.assertEqual(calls[0][2], {"mode": "target dimensions", "width": 128, "height": 64})
        self.assertEqual(calls[0][3:], (32, "cpu", "fp16"))

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


if __name__ == "__main__":
    unittest.main()
