import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeNodeOutput:
    def __init__(self, value):
        self.args = (value,)


def load_advanced_module():
    calls = []
    package_name = "director_advanced_test_package"

    package = types.ModuleType(package_name)
    package.__path__ = []
    nodes_package = types.ModuleType(f"{package_name}.nodes")
    nodes_package.__path__ = []

    class BaseDirector:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}, "optional": {"shift_video": ("FLOAT", {}), "shift_audio": ("FLOAT", {})}}

        def execute(self, **kwargs):
            provider = kwargs.get("_segment_model_provider")
            segment_loras = kwargs.pop("test_segment_loras", None)
            if provider is not None and segment_loras is not None:
                kwargs["segment_models"] = [
                    provider(kwargs["model"], types.SimpleNamespace(loras=loras))
                    for loras in segment_loras
                ]
            calls.append(("director", kwargs["model"]))
            return kwargs

    director_module = types.ModuleType(f"{package_name}.nodes.director")
    director_module.MiniMaxH3Director = BaseDirector

    core_nodes = types.ModuleType("nodes")

    class LoraLoaderModelOnly:
        def load_lora_model_only(self, model, name, strength):
            calls.append(("lora", name, strength))
            return (f"{model}>lora:{name}:{strength}",)

    core_nodes.LoraLoaderModelOnly = LoraLoaderModelOnly
    core_nodes.NODE_CLASS_MAPPINGS = {}

    comfy_extras = types.ModuleType("comfy_extras")
    comfy_extras.__path__ = []
    easycache_module = types.ModuleType("comfy_extras.nodes_easycache")

    class EasyCacheNode:
        @classmethod
        def execute(cls, model, threshold, start, end, verbose):
            calls.append(("easycache", threshold, start, end, verbose))
            return FakeNodeOutput(f"{model}>easycache")

    easycache_module.EasyCacheNode = EasyCacheNode

    firstblock_module = types.ModuleType(f"{package_name}.nodes.first_block_cache")

    def apply_first_block_cache(model, threshold, verbose):
        calls.append(("firstblock", threshold, verbose))
        return f"{model}>firstblock"

    firstblock_module.apply_first_block_cache = apply_first_block_cache

    module_name = f"{package_name}.nodes.director_advanced"
    modules = {
        package_name: package,
        f"{package_name}.nodes": nodes_package,
        f"{package_name}.nodes.director": director_module,
        f"{package_name}.nodes.first_block_cache": firstblock_module,
        module_name: None,
        "nodes": core_nodes,
        "comfy_extras": comfy_extras,
        "comfy_extras.nodes_easycache": easycache_module,
    }
    modules.pop(module_name)
    module_patch = patch.dict(sys.modules, modules)
    module_patch.start()

    module_path = Path(__file__).parents[1] / "nodes" / "director_advanced.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    def cleanup():
        sys.modules.pop(module_name, None)
        module_patch.stop()

    return module, core_nodes, calls, cleanup


class AdvancedDirectorTest(unittest.TestCase):
    def setUp(self):
        self.module, self.core_nodes, self.calls, cleanup = load_advanced_module()
        self.addCleanup(cleanup)

    def test_lora_configuration_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            self.module._parse_loras("{}")
        with self.assertRaisesRegex(ValueError, "needs a filename"):
            self.module._parse_loras('[{"enabled": true, "strength": 1}]')
        with self.assertRaisesRegex(ValueError, "between -100 and 100"):
            self.module._parse_loras('[{"name": "bad.safetensors", "strength": 101}]')

    def test_sampling_shift_inputs_remain_owned_by_the_base_director(self):
        optional = self.module.MiniMaxH3DirectorAdvanced.INPUT_TYPES()["optional"]

        self.assertIn("shift_video", optional)
        self.assertIn("shift_audio", optional)
        self.assertNotIn("enable_model_sampling", optional)
        self.assertEqual(optional["enable_firstblock_cache"][1]["default"], False)
        self.assertEqual(optional["firstblock_cache_threshold"][1]["default"], 0.08)
        self.assertEqual(optional["firstblock_cache_verbose"][1]["default"], False)
        names = list(optional)
        self.assertGreater(names.index("enable_firstblock_cache"), names.index("easycache_verbose"))

    def test_disabled_enhancements_pass_the_original_model_to_director(self):
        result = self.module.MiniMaxH3DirectorAdvanced().execute(
            model="base",
            video_vae="video",
            audio_vae="audio",
            clip="clip",
            shift_video=12,
            shift_audio=3,
        )

        self.assertEqual(result["model"], "base")
        self.assertEqual(result["shift_video"], 12)
        self.assertEqual(result["shift_audio"], 3)
        self.assertEqual(self.calls, [("director", "base")])

    def test_enabled_enhancements_apply_in_model_preparation_order(self):
        calls = self.calls

        class SagePatch:
            def patch(self, model, mode, allow_compile):
                calls.append(("sage", mode, allow_compile))
                return (f"{model}>sage",)

        class MemoryPatch:
            @classmethod
            def execute(cls, model):
                calls.append(("memory",))
                return FakeNodeOutput(f"{model}>memory")

        class SolPatch:
            @classmethod
            def execute(cls, model, *args, **kwargs):
                calls.append(("sol", args, kwargs))
                return FakeNodeOutput(f"{model}>sol")

        self.core_nodes.NODE_CLASS_MAPPINGS.update(
            {
                "PathchSageAttentionKJ": SagePatch,
                "MiniMaxH3MemoryEfficientSageAttentionPatch": MemoryPatch,
                "SolAttnPatch": SolPatch,
            }
        )

        result = self.module.MiniMaxH3DirectorAdvanced().execute(
            model="base",
            video_vae="video",
            audio_vae="audio",
            clip="clip",
            lora_config='[{"name":"one.safetensors","strength":0.75},{"name":"two.safetensors","strength":-0.5}]',
            enable_sage_attention=True,
            sage_attention="auto",
            allow_sage_compile=True,
            enable_h3_mem_eff_sage=True,
            enable_sol_attn=True,
            enable_easycache=True,
            shift_video=10,
            shift_audio=2,
        )

        self.assertTrue(result["model"].endswith(">sage>memory>sol>easycache"))
        self.assertEqual(
            [call[0] for call in self.calls],
            ["lora", "lora", "sage", "memory", "sol", "easycache", "director"],
        )
        self.assertEqual(self.calls[0], ("lora", "one.safetensors", 0.75))
        self.assertEqual(self.calls[1], ("lora", "two.safetensors", -0.5))

    def test_firstblock_cache_applies_after_attention_patch(self):
        calls = self.calls

        class SolPatch:
            @classmethod
            def execute(cls, model, *args, **kwargs):
                calls.append(("sol", args, kwargs))
                return FakeNodeOutput(f"{model}>sol")

        self.core_nodes.NODE_CLASS_MAPPINGS["SolAttnPatch"] = SolPatch

        result = self.module.MiniMaxH3DirectorAdvanced().execute(
            model="base",
            video_vae="video",
            audio_vae="audio",
            clip="clip",
            enable_sol_attn=True,
            enable_firstblock_cache=True,
            firstblock_cache_threshold=0.08,
            firstblock_cache_verbose=True,
        )

        self.assertEqual(result["model"], "base>sol>firstblock")
        self.assertEqual(
            [call[0] for call in self.calls],
            ["sol", "firstblock", "director"],
        )
        self.assertEqual(self.calls[1], ("firstblock", 0.08, True))

    def test_firstblock_cache_rejects_easycache_combination(self):
        with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
            self.module.MiniMaxH3DirectorAdvanced().execute(
                model="base",
                video_vae="video",
                audio_vae="audio",
                clip="clip",
                enable_firstblock_cache=True,
                enable_easycache=True,
            )

        self.assertEqual(self.calls, [])

    def test_firstblock_cache_rejects_legacy_shifted_types(self):
        with self.assertRaisesRegex(ValueError, "enabled must be a Boolean"):
            self.module.MiniMaxH3DirectorAdvanced().execute(
                model="base",
                video_vae="video",
                audio_vae="audio",
                clip="clip",
                enable_firstblock_cache="Performance",
            )

        with self.assertRaisesRegex(ValueError, "threshold must be a finite number"):
            self.module.MiniMaxH3DirectorAdvanced().execute(
                model="base",
                video_vae="video",
                audio_vae="audio",
                clip="clip",
                firstblock_cache_threshold=True,
            )

        with self.assertRaisesRegex(ValueError, "verbose must be a Boolean"):
            self.module.MiniMaxH3DirectorAdvanced().execute(
                model="base",
                video_vae="video",
                audio_vae="audio",
                clip="clip",
                firstblock_cache_verbose="false",
            )

        self.assertEqual(self.calls, [])

    def test_prompt_group_loras_are_isolated_and_follow_shared_loras(self):
        result = self.module.MiniMaxH3DirectorAdvanced().execute(
            model="base",
            video_vae="video",
            audio_vae="audio",
            clip="clip",
            lora_config='[{"name":"shared.safetensors","strength":0.5}]',
            test_segment_loras=[
                [{"name": "first.safetensors", "strength": 0.75}],
                [{"name": "second.safetensors", "strength": -0.25}],
                [],
            ],
        )

        self.assertEqual(
            result["segment_models"],
            [
                "base>lora:shared.safetensors:0.5>lora:first.safetensors:0.75",
                "base>lora:shared.safetensors:0.5>lora:second.safetensors:-0.25",
                "base>lora:shared.safetensors:0.5",
            ],
        )
        self.assertEqual(
            [call for call in self.calls if call[0] == "lora"],
            [
                ("lora", "shared.safetensors", 0.5),
                ("lora", "first.safetensors", 0.75),
                ("lora", "second.safetensors", -0.25),
            ],
        )

    def test_enabled_missing_custom_node_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "PathchSageAttentionKJ is enabled"):
            self.module.MiniMaxH3DirectorAdvanced().execute(
                model="base",
                video_vae="video",
                audio_vae="audio",
                clip="clip",
                enable_sage_attention=True,
            )


if __name__ == "__main__":
    unittest.main()
