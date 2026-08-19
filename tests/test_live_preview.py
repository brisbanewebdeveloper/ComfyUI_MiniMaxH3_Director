import unittest
from unittest.mock import patch

import torch
from PIL import Image

from ComfyUI_MiniMaxH3_Director.director import tae_preview


class LivePreviewTest(unittest.TestCase):
    def test_temporal_indices_cover_the_shot_with_a_cap(self):
        self.assertEqual(tae_preview._temporal_indices(9, 4), [0, 3, 5, 8])
        self.assertEqual(tae_preview._temporal_indices(3, 8), [0, 1, 2])
        self.assertEqual(tae_preview._temporal_indices(9, 1), [4])

    def test_latent_fallback_returns_animated_frame_sequence(self):
        video = torch.zeros((1, 24, 7, 8, 8))

        def fake_decode(_video, temporal_index=None):
            return Image.new("RGB", (32, 24), (int(temporal_index or 0), 0, 0))

        with (
            patch.object(tae_preview, "get_tae_decoder", return_value=None),
            patch.object(tae_preview, "_latent2rgb_pil", side_effect=fake_decode) as decode,
        ):
            frames = tae_preview.x0_to_preview_frames(video, max_side=64, max_frames=4)

        self.assertEqual(len(frames), 4)
        self.assertEqual([call.args[1] for call in decode.call_args_list], [0, 2, 4, 6])
        self.assertTrue(all(max(frame.size) <= 64 for frame in frames))


if __name__ == "__main__":
    unittest.main()
