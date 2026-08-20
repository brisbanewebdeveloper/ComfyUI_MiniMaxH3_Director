import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_cache_module():
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_patcher = types.ModuleType("comfy.model_patcher")
    model_patcher.create_model_options_clone = copy.deepcopy
    patcher_extension = types.ModuleType("comfy.patcher_extension")
    patcher_extension.WrappersMP = types.SimpleNamespace(OUTER_SAMPLE="outer_sample")
    comfy.model_patcher = model_patcher
    comfy.patcher_extension = patcher_extension

    module_name = "first_block_cache_test_module"
    module_patch = patch.dict(
        sys.modules,
        {
            "comfy": comfy,
            "comfy.model_patcher": model_patcher,
            "comfy.patcher_extension": patcher_extension,
        },
    )
    module_patch.start()
    module_path = Path(__file__).parents[1] / "nodes" / "first_block_cache.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    def cleanup():
        sys.modules.pop(module_name, None)
        module_patch.stop()

    return module, cleanup


class MiniMaxH3Model:
    def __init__(self, block_count=3):
        self.blocks = [object() for _ in range(block_count)]


class FakeModel:
    def __init__(self, diffusion_model=None):
        self.model = types.SimpleNamespace(diffusion_model=diffusion_model or MiniMaxH3Model())
        self.model_options = {"transformer_options": {}}
        self.wrappers = []

    def clone(self):
        return copy.deepcopy(self)

    def set_model_patch_replace(self, replacement, name, block_name, index):
        patches = self.model_options["transformer_options"].setdefault("patches_replace", {})
        patches.setdefault(name, {})[(block_name, index)] = replacement

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.append((wrapper_type, key, wrapper))

    def get_wrappers(self, wrapper_type, key):
        return [
            wrapper
            for current_type, current_key, wrapper in self.wrappers
            if current_type == wrapper_type and current_key == key
        ]


class FirstBlockCacheTest(unittest.TestCase):
    def setUp(self):
        self.module, cleanup = load_cache_module()
        self.addCleanup(cleanup)

    def test_cache_reuses_tail_below_threshold_and_refreshes_above_it(self):
        model = self.module.apply_first_block_cache(FakeModel(), threshold=0.08)
        options = model.model_options["transformer_options"]
        options["uuids"] = ["conditioning"]
        replacements = options["patches_replace"]["dit"]
        calls = [0, 0, 0]

        def original(index):
            def run(args):
                calls[index] += 1
                if index == 0:
                    return {"img": args["img"] * 2.0}
                return {"img": args["img"] + (10.0 if index == 1 else 100.0)}

            return run

        def run_stack(value):
            hidden = torch.tensor([value])
            for index in range(3):
                hidden = replacements[("double_block", index)](
                    {"img": hidden, "transformer_options": options},
                    {"original_block": original(index)},
                )["img"]
            return hidden

        self.assertTrue(torch.equal(run_stack(1.0), torch.tensor([112.0])))
        self.assertTrue(torch.allclose(run_stack(1.04), torch.tensor([112.08])))
        self.assertTrue(torch.allclose(run_stack(1.2), torch.tensor([112.4])))
        self.assertEqual(calls, [3, 2, 2])

        holder = options["minimax_h3_firstblock_cache"]
        self.assertEqual((holder.calls, holder.compute, holder.reuse), (3, 2, 1))

    def test_cache_measures_residual_when_first_block_updates_input_in_place(self):
        model = self.module.apply_first_block_cache(FakeModel(), threshold=0.08)
        options = model.model_options["transformer_options"]
        replacements = options["patches_replace"]["dit"]
        calls = [0, 0, 0]

        def original(index):
            def run(args):
                calls[index] += 1
                if index == 0:
                    args["img"].mul_(2.0)
                else:
                    args["img"].add_(10.0 if index == 1 else 100.0)
                return {"img": args["img"]}

            return run

        def run_stack(value):
            hidden = torch.tensor([value])
            for index in range(3):
                hidden = replacements[("double_block", index)](
                    {"img": hidden, "transformer_options": options},
                    {"original_block": original(index)},
                )["img"]
            return hidden

        self.assertTrue(torch.equal(run_stack(1.0), torch.tensor([112.0])))
        self.assertTrue(torch.allclose(run_stack(1.04), torch.tensor([112.08])))
        self.assertEqual(calls, [2, 1, 1])

    def test_cache_state_is_isolated_by_conditioning_key(self):
        holder = self.module.FirstBlockCacheHolder(0.08)
        first = {"uuids": ["first"]}
        second = {"uuids": ["second"]}

        head = torch.tensor([2.0])
        _, reused = holder.after_head(torch.tensor([1.0]), head, first)
        self.assertFalse(reused)
        holder.after_tail(torch.tensor([5.0]), first)

        _, reused = holder.after_head(torch.tensor([1.0]), head, second)
        self.assertFalse(reused)
        self.assertEqual((holder.compute, holder.reuse), (2, 0))

    def test_shape_change_forces_refresh(self):
        holder = self.module.FirstBlockCacheHolder(0.08)
        options = {}
        holder.after_head(torch.ones(1), torch.full((1,), 2.0), options)
        holder.after_tail(torch.full((1,), 5.0), options)

        _, reused = holder.after_head(torch.ones(2), torch.full((2,), 2.0), options)

        self.assertFalse(reused)
        self.assertEqual((holder.compute, holder.reuse), (2, 0))

    def test_refresh_releases_obsolete_tail_before_running_blocks(self):
        holder = self.module.FirstBlockCacheHolder(0.08)
        options = {}
        holder.after_head(torch.ones(1), torch.full((1,), 2.0), options)
        holder.after_tail(torch.full((1,), 5.0), options)

        _, reused = holder.after_head(torch.ones(1), torch.full((1,), 3.0), options)

        self.assertFalse(reused)
        state = holder.state_for(options)
        self.assertIsNone(state.tail_residual)
        self.assertIsNotNone(state.pending_head_output)

    def test_large_cache_is_bypassed_and_runs_the_full_stack(self):
        model = self.module.apply_first_block_cache(FakeModel(), threshold=0.08)
        options = model.model_options["transformer_options"]
        replacements = options["patches_replace"]["dit"]
        calls = [0, 0, 0]

        def original(index):
            def run(args):
                calls[index] += 1
                return {"img": args["img"] * 2.0 if index == 0 else args["img"] + 10.0}

            return run

        def run_stack(value):
            hidden = torch.tensor([value])
            for index in range(3):
                hidden = replacements[("double_block", index)](
                    {"img": hidden, "transformer_options": options},
                    {"original_block": original(index)},
                )["img"]
            return hidden

        with self.assertLogs(self.module._log, level="WARNING") as logs:
            with patch.object(self.module, "_MAX_RETAINED_CACHE_BYTES", 1):
                first = run_stack(1.0)
                second = run_stack(1.04)

        self.assertTrue(torch.equal(first, torch.tensor([22.0])))
        self.assertTrue(torch.allclose(second, torch.tensor([22.08])))
        self.assertEqual(calls, [2, 2, 2])
        self.assertIn("FirstBlockCache bypassed for this execution", logs.output[0])
        holder = options["minimax_h3_firstblock_cache"]
        self.assertEqual((holder.calls, holder.compute, holder.reuse), (2, 2, 0))
        state = holder.state_for(options)
        self.assertTrue(state.cache_disabled)
        self.assertIsNone(state.previous_head_residual)
        self.assertIsNone(state.tail_residual)
        self.assertIsNone(state.pending_head_output)

    def test_device_pressure_bypasses_cache_below_required_headroom(self):
        hidden = types.SimpleNamespace(
            numel=lambda: 1,
            element_size=lambda: 2,
            device=torch.device("cuda"),
        )

        with patch.object(torch.cuda, "mem_get_info", return_value=(3 * 1024**3, 24 * 1024**3)):
            reason = self.module._cache_bypass_reason(hidden)

        self.assertIn("device headroom", reason)

    def test_install_preserves_an_existing_block_replacement(self):
        model = FakeModel()
        calls = []

        def previous(args, extra_options):
            calls.append("previous")
            result = extra_options["original_block"](args)
            return {"img": result["img"] + 1.0}

        model.model_options["transformer_options"]["patches_replace"] = {
            "dit": {("double_block", 1): previous}
        }
        patched = self.module.apply_first_block_cache(model)
        options = patched.model_options["transformer_options"]
        replacement = options["patches_replace"]["dit"][("double_block", 1)]
        state = options["minimax_h3_firstblock_cache"].state_for(options)
        state.skip_tail = False

        result = replacement(
            {"img": torch.tensor([1.0]), "transformer_options": options},
            {"original_block": lambda args: {"img": args["img"] + 2.0}},
        )

        self.assertEqual(calls, ["previous"])
        self.assertTrue(torch.equal(result["img"], torch.tensor([4.0])))

    def test_sampling_wrapper_restores_options_after_failure(self):
        model = self.module.apply_first_block_cache(FakeModel())
        wrapper = model.wrappers[0][2]
        guider = types.SimpleNamespace(model_options=model.model_options)
        original_options = guider.model_options

        class Executor:
            class_obj = guider

            def __call__(self, *args, **kwargs):
                holder = guider.model_options["transformer_options"]["minimax_h3_firstblock_cache"]
                holder.calls = 1
                raise RuntimeError("sampling failed")

        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            wrapper(Executor())

        self.assertIs(guider.model_options, original_options)
        original_holder = original_options["transformer_options"]["minimax_h3_firstblock_cache"]
        self.assertEqual(original_holder.calls, 0)

    def test_install_rejects_wrong_model_and_easycache_conflict(self):
        wrong = FakeModel(types.SimpleNamespace(blocks=[object(), object()]))
        with self.assertRaisesRegex(ValueError, "requires a MiniMax H3"):
            self.module.apply_first_block_cache(wrong)

        model = FakeModel()
        model.model_options["transformer_options"]["easycache"] = object()
        with self.assertRaisesRegex(ValueError, "cannot be combined with EasyCache"):
            self.module.apply_first_block_cache(model)

    def test_install_rejects_spectrum_wrapper(self):
        model = FakeModel()
        model.add_wrapper_with_key("outer_sample", "spectrum_minimax_h3", object())

        self.assertTrue(self.module.has_spectrum_wrapper(model))
        with self.assertRaisesRegex(ValueError, "Spectrum and FirstBlockCache cannot both be enabled"):
            self.module.apply_first_block_cache(model)

    def test_threshold_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self.module.FirstBlockCacheHolder(0.0)
        with self.assertRaisesRegex(ValueError, "at most one"):
            self.module.FirstBlockCacheHolder(1.01)


if __name__ == "__main__":
    unittest.main()
