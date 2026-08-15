import types
import unittest
from unittest import mock

from director import h3_context_patches


def layout_module(init):
    class PackedLayout:
        pass

    PackedLayout.__init__ = init
    return types.SimpleNamespace(PackedLayout=PackedLayout, FRAME_RESCALE=1.0)


class LayoutPatchCompatibilityTest(unittest.TestCase):
    def setUp(self):
        h3_context_patches._layout_orig = None
        h3_context_patches._layout_applied = False

    def tearDown(self):
        h3_context_patches._layout_orig = None
        h3_context_patches._layout_applied = False

    def test_solattn_layout_wrapper_is_composed_with_director(self):
        def solattn_init(self, *args, **kwargs):
            pass

        solattn_init.__module__ = "ComfyUI_SolAttn_triton._morton_h3"
        mm = layout_module(solattn_init)

        with (
            mock.patch.object(h3_context_patches, "_mm", return_value=mm),
            mock.patch.object(h3_context_patches, "_self_test_layout"),
        ):
            self.assertTrue(h3_context_patches.ensure_layout_patch())

        self.assertIs(h3_context_patches._layout_orig, solattn_init)
        self.assertIs(mm.PackedLayout.__init__, h3_context_patches._director_layout_init)

    def test_unknown_layout_wrapper_remains_rejected(self):
        def unknown_init(self, *args, **kwargs):
            pass

        unknown_init.__module__ = "unknown_h3_pack.layout_patch"
        mm = layout_module(unknown_init)

        with mock.patch.object(h3_context_patches, "_mm", return_value=mm):
            with self.assertRaisesRegex(RuntimeError, "another pack already patched"):
                h3_context_patches.ensure_layout_patch()


if __name__ == "__main__":
    unittest.main()
