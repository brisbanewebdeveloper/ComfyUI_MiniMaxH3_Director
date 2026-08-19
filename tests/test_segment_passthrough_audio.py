import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ComfyUI_MiniMaxH3_Director.director import segment_runtime


class SegmentPassthroughAudioTest(unittest.TestCase):
    def test_extracts_matching_source_range_for_unselected_segment(self):
        plan = SimpleNamespace(raw={"video": {"videoFile": "source.mp4"}}, frame_rate=24)
        segment = SimpleNamespace(task_key="v2v", start_frame=24, end_frame=72)
        expected = {"waveform": "audio", "sample_rate": 44100}

        with patch.object(segment_runtime, "extract_timeline_audio", return_value=expected) as extract:
            result = segment_runtime.segment_passthrough_audio(plan, segment)

        self.assertEqual(result, expected)
        extract.assert_called_once_with(plan.raw, 24, 72, 24.0)

    def test_generation_timeline_never_uses_source_audio_passthrough(self):
        plan = SimpleNamespace(raw={"timelineMode": "prompt_batch"}, frame_rate=24)
        segment = SimpleNamespace(task_key="fl2v", start_frame=0, end_frame=24)

        with patch.object(segment_runtime, "extract_timeline_audio") as extract:
            result = segment_runtime.segment_passthrough_audio(plan, segment)

        self.assertIsNone(result)
        extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
