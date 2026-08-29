import json
import unittest
from unittest.mock import MagicMock, patch

import torch

from ComfyUI_MiniMaxH3_Director.nodes.director import MiniMaxH3Director
from ComfyUI_MiniMaxH3_Director.nodes.enhance_prompt import MiniMaxH3DirectorEnhancePrompt
from ComfyUI_MiniMaxH3_Director.nodes.save_last_frame import MiniMaxH3DirectorSaveLastFrame
from ComfyUI_MiniMaxH3_Director.lib.task_prompts import resolve_task_key, task_type_combo_options


class ConvenienceNodesTest(unittest.TestCase):
    def test_user_facing_schema_defaults_to_english(self):
        task_options, task_meta = task_type_combo_options()
        enhancer_inputs = MiniMaxH3DirectorEnhancePrompt.INPUT_TYPES()
        director_inputs = MiniMaxH3Director.INPUT_TYPES()

        self.assertEqual(task_options[0], "t2v — Text to Video")
        self.assertEqual(task_meta["default"], "t2v — Text to Video")
        self.assertIn("MiniMax H3 supports", task_meta["tooltip"])
        self.assertEqual(director_inputs["required"]["bd_grp_sample"][1]["default"], "Sampling settings")
        self.assertEqual(director_inputs["optional"]["bd_grp_advanced"][1]["default"], "Advanced sampling")
        self.assertEqual(director_inputs["optional"]["bd_grp_perf"][1]["default"], "Performance")
        self.assertEqual(enhancer_inputs["optional"]["openai_compat_mode"][0][0], "Standard")
        self.assertEqual(enhancer_inputs["optional"]["output_language"][0], ["English", "Chinese"])

    def test_legacy_chinese_combo_values_remain_valid(self):
        legacy_task = "r2v — 参考主体生视频(Reference to Video)"

        self.assertEqual(resolve_task_key(legacy_task), "r2v")
        self.assertTrue(
            MiniMaxH3DirectorEnhancePrompt.VALIDATE_INPUTS(
                legacy_task,
                openai_compat_mode="标准",
                output_language="中文",
            )
        )

    def test_director_exposes_lazy_ref_checkpoint(self):
        inputs = MiniMaxH3Director.INPUT_TYPES()

        self.assertTrue(inputs["required"]["model"][1]["lazy"])
        self.assertTrue(inputs["optional"]["model_ref2va"][1]["lazy"])

    def test_lazy_status_only_requests_selected_checkpoint(self):
        ref_timeline = json.dumps(
            {"global": {"taskType": "v2v"}, "segments": [{"taskType": "v2v"}]}
        )
        fl_timeline = json.dumps(
            {"global": {"taskType": "fl2v"}, "segments": [{"taskType": "fl2v"}]}
        )

        self.assertEqual(
            MiniMaxH3Director.check_lazy_status(
                timeline_data=ref_timeline,
                model=None,
                model_ref2va=None,
            ),
            ["model_ref2va"],
        )
        self.assertEqual(
            MiniMaxH3Director.check_lazy_status(
                timeline_data=fl_timeline,
                model=None,
                model_ref2va=None,
            ),
            ["model"],
        )

    def test_enhancer_returns_original_prompt_and_error_on_failure(self):
        with patch(
            "ComfyUI_MiniMaxH3_Director.nodes.enhance_prompt.enhance_prompt_sync",
            return_value=(None, "offline"),
        ):
            output = MiniMaxH3DirectorEnhancePrompt().enhance(
                "A runner crosses the street",
                "t2v — Text to Video",
                "Ollama",
                "http://127.0.0.1:11434",
                "qwen3.5",
            )

        self.assertEqual(output, ("A runner crosses the street", "offline"))

    def test_save_last_frame_delegates_only_final_image(self):
        images = torch.zeros((3, 8, 8, 3))
        saver = MagicMock()
        saver.save_images.return_value = {"result": (images[-1:],)}

        with patch(
            "ComfyUI_MiniMaxH3_Director.nodes.save_last_frame.comfy_nodes.SaveImage",
            return_value=saver,
        ):
            result = MiniMaxH3DirectorSaveLastFrame().save(images, "test/last")

        saved = saver.save_images.call_args.args[0]
        self.assertTrue(torch.equal(saved, images[-1:]))
        self.assertEqual(result["result"][0].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
