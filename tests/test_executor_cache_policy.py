"""Exercise the post-decode cache transaction without model loading."""
import ast
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "director" / "executor_core.py"


def compile_nodes(statements):
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *statements], type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(SOURCE), "exec")


class ExecutorCachePolicyTest(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        policy = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_segment_disk_cache_needed")
        self.globals = {"__package__": "cache_test", "time": time}
        exec(compile_nodes([policy]), self.globals)
        runner = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_one_segment")
        start = next(i for i, n in enumerate(runner.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cpu_samples" for t in n.targets))
        end = next(i for i, n in enumerate(runner.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cache_s" for t in n.targets))
        self.cache_block = compile_nodes(runner.body[start:end + 1])
        face = types.ModuleType("cache_test.face_refine.pack")
        face.face_refine_enabled = lambda plan: bool(plan.face_refine)
        lift = types.ModuleType("cache_test.selflift.pack")
        lift.selflift_enabled = lambda plan: bool(plan.selflift)
        self.modules = {face.__name__: face, lift.__name__: lift}

    def transaction(self, **options):
        plan = types.SimpleNamespace(segments=[object()], continuity_enabled=False, face_refine=False, selflift=False)
        for name in ("continuity_enabled", "face_refine", "selflift"):
            setattr(plan, name, options.pop(name, False))
        cpu = {"samples": object()}
        chunk = types.SimpleNamespace(shape=(124, 1024, 576, 3))
        save = Mock()
        namespace = {**self.globals, "plan": plan, "samples": {"samples": object()},
                     "av_latent_to_cpu": Mock(return_value=cpu), "completed_av_latents": {99: object()},
                     "seg": types.SimpleNamespace(index=0), "chunk": chunk, "handoff": {"trim_frames": 0},
                     "completed_av_handoff": {}, "audio_dict": {"waveform": object()}, "completed_audios": {},
                     "timeline_seg_total": 1, "will_refine": False, "hold_after_first": False,
                     "save_segment_cache": save, "node_id": "5", **options}
        with patch.dict(sys.modules, self.modules):
            exec(self.cache_block, namespace)
        self.assertNotIn("samples", namespace)
        self.assertEqual(namespace["completed_av_latents"], {0: cpu})
        return namespace, save, cpu

    def test_single_shot_skips_cache_but_retains_only_cpu_handoff(self):
        namespace, save, _ = self.transaction()
        self.assertFalse(namespace["write_cache"])
        save.assert_not_called()

    def test_cache_required_paths_save_retained_cpu_latent(self):
        for options in ({"will_refine": True}, {"hold_after_first": True}, {"continuity_enabled": True},
                        {"timeline_seg_total": 2}, {"face_refine": True}, {"selflift": True}):
            with self.subTest(options=options):
                namespace, save, cpu = self.transaction(**options)
                self.assertTrue(namespace["write_cache"])
                save.assert_called_once()
                self.assertIs(save.call_args.kwargs["av_latent"], cpu)
                self.assertIs(save.call_args.kwargs["handoff"], namespace["handoff"])
                self.assertIs(save.call_args.kwargs["audio"], namespace["audio_dict"])


if __name__ == "__main__":
    unittest.main()
