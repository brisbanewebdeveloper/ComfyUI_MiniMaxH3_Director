import base64
import io
import unittest

from PIL import Image

from ComfyUI_MiniMaxH3_Director.director.gen_timeline import build_gen_director_plan
from ComfyUI_MiniMaxH3_Director.director.segment_runtime import resolve_segment_raw_clip


def image_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (32, 64, 96)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def i2v_timeline(
    *,
    continuity_enabled: bool,
    first_image: bool = True,
    second_image: bool = False,
    second_from_prev: bool = True,
) -> dict:
    image = {"imageB64": image_b64(), "width": 32, "height": 32}
    return {
        "version": 5,
        "editMode": "segment",
        "timelineMode": "prompt_batch",
        "frameRate": 24,
        "global": {"taskType": "i2v", "prompt": ""},
        "output": {
            "mode": "fixed",
            "width": 32,
            "height": 32,
            "continuityEnabled": continuity_enabled,
            "continuityOverlapFrames": 22,
        },
        "segments": [
            {
                "frameCount": 4,
                "prompt": "first",
                "genImage": image if first_image else {"imageFile": ""},
            },
            {
                "frameCount": 4,
                "prompt": "second",
                "genImage": image if second_image else {"imageFile": ""},
                "continuityFromPrev": second_from_prev,
            },
        ],
    }


def build_plan(timeline: dict):
    return build_gen_director_plan(
        timeline,
        global_task_type="i2v",
        global_prompt="",
        total_frames=8,
        frame_rate=24,
        width=32,
        height=32,
        ref_max_size=32,
    )


class GenTimelineContinuityTest(unittest.TestCase):
    def test_first_i2v_group_always_requires_source_image(self):
        timeline = i2v_timeline(continuity_enabled=True, first_image=False)

        with self.assertRaisesRegex(ValueError, "first i2v prompt group"):
            build_plan(timeline)

    def test_later_i2v_group_requires_source_when_continuity_is_off(self):
        timeline = i2v_timeline(continuity_enabled=False)

        with self.assertRaisesRegex(ValueError, "Segment #2.*Segment continuity"):
            build_plan(timeline)

    def test_later_i2v_group_requires_source_when_from_prev_is_off(self):
        timeline = i2v_timeline(
            continuity_enabled=True,
            second_from_prev=False,
        )

        with self.assertRaisesRegex(ValueError, "Segment #2.*From prev"):
            build_plan(timeline)

    def test_later_i2v_group_can_use_previous_segment_as_source(self):
        plan = build_plan(i2v_timeline(continuity_enabled=True))

        self.assertTrue(plan.continuity_enabled)
        self.assertIsNotNone(plan.segments[0].source_clip)
        self.assertIsNone(plan.segments[1].source_clip)
        self.assertTrue(plan.segments[1].continuity_from_prev)
        self.assertEqual(plan.source_video.shape[0], 2)
        self.assertEqual(resolve_segment_raw_clip(plan, plan.segments[1]).shape[0], 0)

    def test_explicit_later_source_remains_a_fresh_anchor(self):
        plan = build_plan(
            i2v_timeline(continuity_enabled=True, second_image=True)
        )

        self.assertIsNotNone(plan.segments[1].source_clip)
        self.assertTrue(plan.segments[1].continuity_from_prev)

    def test_prompt_group_loras_are_preserved_on_the_segment_plan(self):
        timeline = i2v_timeline(continuity_enabled=True)
        timeline["segments"][1]["loras"] = [
            {"enabled": True, "name": "detail.safetensors", "strength": 0.8}
        ]

        plan = build_plan(timeline)

        self.assertEqual(plan.segments[0].loras, [])
        self.assertEqual(plan.segments[1].loras, timeline["segments"][1]["loras"])


if __name__ == "__main__":
    unittest.main()
