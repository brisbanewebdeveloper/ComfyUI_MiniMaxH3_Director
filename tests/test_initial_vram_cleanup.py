import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ComfyUI_MiniMaxH3_Director.director import executor_core


class ConditioningReached(RuntimeError):
    pass


def _execution_plan() -> SimpleNamespace:
    segment = SimpleNamespace(
        index=0,
        timeline_index=0,
        task_key="t2v",
        start_frame=0,
        end_frame=5,
        frame_count=5,
        source_clip=None,
        prompt="test",
        refs=[],
        ref_videos=[],
        ref_audios=[],
    )
    return SimpleNamespace(
        raw={},
        segments=[segment],
        run_indices=None,
        continuity_enabled=False,
        continuity_overlap_frames=0,
        total_frames=5,
        output_mode="fixed",
        width=32,
        height=32,
        ref_max_size=32,
    )


class InitialVramCleanupTest(unittest.TestCase):
    def run_until_conditioning(self, clear_vram_between_segments: bool) -> list[str]:
        events: list[str] = []

        def cleanup(*, enabled=True, unload_models=True):
            events.append("cleanup")

        def conditioning(**kwargs):
            events.append("conditioning")
            raise ConditioningReached

        empty_clip = torch.empty((0, 32, 32, 3))
        with (
            patch.object(executor_core, "cleanup_segment_vram", side_effect=cleanup),
            patch.object(executor_core, "plan_summary", return_value="test plan"),
            patch.object(executor_core, "new_segment_mp4_run_dir", return_value=None),
            patch.object(executor_core, "report_director_progress"),
            patch.object(executor_core, "resolve_segment_raw_clip", return_value=empty_clip),
            patch.object(executor_core, "compile_reference_prompt", return_value="test"),
            patch.object(executor_core, "reference_semantics_for_segment", return_value=None),
            patch.object(executor_core, "_build_minimax_inputs", return_value=(None,) * 6),
            patch.object(executor_core, "run_minimax_conditioning", side_effect=conditioning),
        ):
            with self.assertRaises(ConditioningReached):
                executor_core.execute_director_plan_core(
                    _execution_plan(),
                    model=object(),
                    vae=object(),
                    audio_vae=object(),
                    clip=object(),
                    clear_vram_between_segments=clear_vram_between_segments,
                )

        return events

    def test_enabled_cleanup_runs_before_first_conditioning(self):
        self.assertEqual(self.run_until_conditioning(True), ["cleanup", "conditioning"])

    def test_disabled_cleanup_does_not_run_before_conditioning(self):
        self.assertEqual(self.run_until_conditioning(False), ["conditioning"])


if __name__ == "__main__":
    unittest.main()
