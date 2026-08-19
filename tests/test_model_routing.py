import json
import unittest

from ComfyUI_MiniMaxH3_Director.director.model_routing import (
    FL2VA_FAMILY,
    REF2VA_FAMILY,
    model_for_segment,
    timeline_model_families,
)


class ModelRoutingTest(unittest.TestCase):
    def test_selected_segments_determine_loaded_families(self):
        timeline = {
            "global": {"taskType": "v2v — Video to Video"},
            "runSelectEnabled": True,
            "runSelection": [1],
            "segments": [
                {"taskType": "v2v — Video to Video"},
                {"taskType": "fl2v — First-Last Frame"},
            ],
        }

        families = timeline_model_families(json.dumps(timeline), "v2v")

        self.assertEqual(families, frozenset({FL2VA_FAMILY}))

    def test_mixed_timeline_requires_both_families(self):
        timeline = {
            "global": {"taskType": "v2v"},
            "segments": [{"taskType": "fl2v"}, {"taskType": "rv2v"}],
        }

        self.assertEqual(
            timeline_model_families(json.dumps(timeline), "v2v"),
            frozenset({FL2VA_FAMILY, REF2VA_FAMILY}),
        )

    def test_model_selection_prefers_family_and_falls_back(self):
        self.assertEqual(model_for_segment("fl", "ref", "fl2v"), "fl")
        self.assertEqual(model_for_segment("fl", "ref", "r2v"), "ref")
        self.assertEqual(model_for_segment("fl", None, "r2v"), "fl")
        self.assertEqual(model_for_segment(None, "ref", "t2v"), "ref")


if __name__ == "__main__":
    unittest.main()
