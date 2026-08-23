import unittest

import torch
from comfy.nested_tensor import NestedTensor

from ComfyUI_MiniMaxH3_Director.director import tiled_refine


def settings(**overrides):
    values = {
        "tiled_refine_temporal": True,
        "tiled_refine_chunk_length": 85,
        "tiled_refine_temporal_overlap": 17,
        "tiled_refine_spatial": True,
        "tiled_refine_tile_width": 320,
        "tiled_refine_tile_height": 320,
        "tiled_refine_tile_overlap": 64,
        "tiled_refine_fade": 32,
        "tiled_refine_min_tile_size": 160,
        "tiled_refine_overlap_mode": "earlier",
        "tiled_refine_blend": "linear",
    }
    values.update(overrides)
    return values


class TiledRefineTest(unittest.TestCase):
    def test_temporal_windows_bound_long_segment_with_cycle_aligned_overlap(self):
        windows = tiled_refine.temporal_windows(122, 85, 17)

        self.assertEqual(len(windows), 6)
        self.assertEqual(windows[0], (0, 25, 0, 85))
        self.assertEqual(windows[1], (20, 45, 68, 153))
        self.assertEqual(windows[-1][1], 122)

    def test_temporal_windows_reject_non_grid_values(self):
        with self.assertRaisesRegex(ValueError, "multiples of 17"):
            tiled_refine.temporal_windows(30, 84, 17)
        with self.assertRaisesRegex(ValueError, "smaller than chunk length"):
            tiled_refine.temporal_windows(30, 85, 85)

    def test_spatial_grid_splits_failed_refine_canvas_into_four_tiles(self):
        grid = tiled_refine.spatial_grid(
            86,
            48,
            settings(
                tiled_refine_tile_width=480,
                tiled_refine_tile_height=864,
                tiled_refine_tile_overlap=128,
                tiled_refine_min_tile_size=256,
            ),
        )

        self.assertEqual(len(grid), 4)
        self.assertEqual(grid[0], (0, 54, 0, 30, 0, 0))
        self.assertEqual(grid[-1], (46, 86, 22, 48, 8, 8))

    def test_conditioning_is_rebased_and_target_keyframe_is_cropped(self):
        keyframe = torch.ones((1, 24, 1, 20, 30))
        conditioning = [
            [
                torch.zeros((1, 2, 3)),
                {
                    "minimax_keyframes": [
                        {"resolved_frame_index": 68, "latent": keyframe},
                        {"resolved_frame_index": 200, "latent": keyframe},
                    ]
                },
            ]
        ]

        result = tiled_refine.conditioning_for_piece(
            conditioning,
            frame_start=68,
            frame_end=153,
            source_height=20,
            source_width=30,
            tile=(0, 10, 5, 15),
        )

        metadata = result[0][1]
        self.assertEqual(metadata["minimax_frame_count"], 85)
        self.assertEqual(len(metadata["minimax_keyframes"]), 1)
        self.assertEqual(metadata["minimax_keyframes"][0]["resolved_frame_index"], 0)
        self.assertEqual(tuple(metadata["minimax_keyframes"][0]["latent"].shape[-2:]), (10, 10))

    def test_sampling_is_bounded_to_pieces_and_preserves_audio(self):
        video = torch.zeros((1, 24, 30, 20, 30))
        audio = torch.zeros((1, 32, 2, 102))
        latent = {"samples": NestedTensor((video, audio))}
        calls = []
        progress = []

        def sample_piece(piece, positive, negative, seed):
            piece_video, piece_audio = piece["samples"].unbind()
            calls.append((tuple(piece_video.shape), seed, positive, negative))
            return {"samples": NestedTensor((piece_video + 1, piece_audio + 100))}

        result = tiled_refine.sample_tiled_refine(
            latent,
            [],
            [],
            settings(),
            seed=7,
            sample_piece=sample_piece,
            on_progress=lambda done, total: progress.append((done, total)),
        )

        result_video, result_audio = result["samples"].unbind()
        self.assertEqual(len(calls), 4)
        self.assertEqual([call[1] for call in calls], [7, 8, 9, 10])
        self.assertEqual(progress[-1], (4, 4))
        self.assertGreater(float(result_video.max()), 0.0)
        self.assertIs(result_audio, audio)
        self.assertEqual(float(result_audio.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
