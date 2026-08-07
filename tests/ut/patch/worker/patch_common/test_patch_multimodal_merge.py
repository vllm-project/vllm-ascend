#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import importlib
import sys
import types

import torch
import vllm.model_executor.models.utils as vllm_models_utils

from tests.ut.base import TestBase


class TestPatchMultimodalMerge(TestBase):
    """
    Tests for ``vllm_ascend.patch.worker.patch_multimodal_merge``.

    The patch must:

    1. Replace ``utils._merge_multimodal_embeddings`` so that any module
       imported AFTER the patch sees the new function.
    2. Propagate the replacement to model modules that already imported
       the original function via ``from .utils import _merge_multimodal_embeddings``,
       so that pre-imported callers (the common case in production) actually
       use the patched version.
    3. Be functionally equivalent on the merge semantics: writing the
       multimodal embeddings into ``inputs_embeds`` at masked positions.
    4. Tolerate / preserve bindings that were deliberately replaced by
       another patch (only update bindings still pointing to the original
       function).
    """

    PATCH_MODULE = "vllm_ascend.patch.worker.patch_multimodal_merge"
    FAKE_MODULE_PREFIX = "vllm.model_executor.models._test_patch_mm_merge_"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reload_patch(self):
        """Re-execute the patch module so that it captures the *current*
        ``utils._merge_multimodal_embeddings`` as its ``orig_merge_mm`` and
        re-runs the sys.modules sweep."""
        if self.PATCH_MODULE in sys.modules:
            return importlib.reload(sys.modules[self.PATCH_MODULE])
        return importlib.import_module(self.PATCH_MODULE)

    def _make_fake_model_module(self, name_suffix, fn):
        """Register a fake model module that holds ``fn`` as its local
        ``_merge_multimodal_embeddings`` binding, simulating a model file
        that did ``from .utils import _merge_multimodal_embeddings``."""
        full_name = self.FAKE_MODULE_PREFIX + name_suffix
        mod = types.ModuleType(full_name)
        mod._merge_multimodal_embeddings = fn  # type: ignore[attr-defined]
        sys.modules[full_name] = mod
        self.addCleanup(sys.modules.pop, full_name, None)
        return mod

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_utils_attribute_is_replaced(self):
        """utils._merge_multimodal_embeddings must point to the patched
        implementation after the patch runs."""
        patch_mod = self._reload_patch()
        self.assertIs(
            vllm_models_utils._merge_multimodal_embeddings,
            getattr(patch_mod, "_merge_multimodal_embeddings"),
        )

    def test_propagates_to_already_imported_model_module(self):
        """A fake model module that imported the ORIGINAL function before the
        patch ran should have its local binding updated by the sweep."""
        # Simulate state before patch: fake module holds whatever utils
        # currently exposes.
        original_fn = vllm_models_utils._merge_multimodal_embeddings
        fake_mod = self._make_fake_model_module("propagate", original_fn)

        # Re-run the patch; sweep should pick up the fake module.
        patch_mod = self._reload_patch()

        self.assertIs(
            getattr(fake_mod, "_merge_multimodal_embeddings"),
            getattr(patch_mod, "_merge_multimodal_embeddings"),
            "fake model module's local _merge_multimodal_embeddings should "
            "be updated to the patched version by the sys.modules sweep",
        )

    def test_does_not_overwrite_unrelated_bindings(self):
        """Modules whose ``_merge_multimodal_embeddings`` is NOT the original
        function (e.g. replaced by another patch, or simply unrelated)
        must NOT be overwritten."""

        def unrelated_fn(*args, **kwargs):  # pragma: no cover - sentinel
            raise AssertionError("should not be called")

        fake_mod = self._make_fake_model_module("unrelated", unrelated_fn)

        self._reload_patch()

        self.assertIs(
            getattr(fake_mod, "_merge_multimodal_embeddings"),
            unrelated_fn,
            "patch should leave bindings unrelated to the original function "
            "untouched",
        )

    def test_module_outside_models_namespace_is_skipped(self):
        """Only modules under ``vllm.model_executor.models`` are considered.
        A module outside that namespace must not be touched even if it
        happens to hold the original function reference."""
        original_fn = vllm_models_utils._merge_multimodal_embeddings

        outside_name = "vllm.unrelated.test_outside_namespace"
        outside_mod = types.ModuleType(outside_name)
        outside_mod._merge_multimodal_embeddings = original_fn  # type: ignore[attr-defined]
        sys.modules[outside_name] = outside_mod
        self.addCleanup(sys.modules.pop, outside_name, None)

        self._reload_patch()

        # Should still be the (original or already-patched) function we put
        # in, not the new patch function selected by namespace sweep.
        self.assertIs(
            getattr(outside_mod, "_merge_multimodal_embeddings"),
            original_fn,
            "patch should not sweep modules outside vllm.model_executor.models",
        )

    def test_merge_writes_at_mask_positions(self):
        """Functional check: the patched ``_merge_multimodal_embeddings``
        must write multimodal embeddings into the True positions of the
        mask, leaving other positions unchanged."""
        from vllm_ascend.patch.worker.patch_multimodal_merge import (
            _merge_multimodal_embeddings,
        )

        n_tokens, hidden = 6, 4
        inputs_embeds = torch.zeros(n_tokens, hidden, dtype=torch.float32)
        # Mark positions 1, 2, 4 as multimodal (3 placeholders).
        is_multimodal = torch.tensor(
            [False, True, True, False, True, False], dtype=torch.bool
        )
        # Provide 3 multimodal embeddings to fill those slots.
        mm_embeddings = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ],
            dtype=torch.float32,
        )

        out = _merge_multimodal_embeddings(
            inputs_embeds, [mm_embeddings], is_multimodal
        )

        # In-place: returned tensor is the same object.
        self.assertIs(out, inputs_embeds)
        # True positions should hold mm_embeddings rows in order.
        self.assertTrue(torch.equal(inputs_embeds[1], mm_embeddings[0]))
        self.assertTrue(torch.equal(inputs_embeds[2], mm_embeddings[1]))
        self.assertTrue(torch.equal(inputs_embeds[4], mm_embeddings[2]))
        # False positions should still be zero.
        for false_idx in (0, 3, 5):
            self.assertTrue(
                torch.equal(
                    inputs_embeds[false_idx], torch.zeros(hidden)
                )
            )

    def test_merge_count_mismatch_raises_value_error(self):
        """If the count of multimodal placeholders differs from the number
        of provided multimodal embeddings, the function must raise
        ``ValueError`` (not let a generic RuntimeError leak)."""
        from vllm_ascend.patch.worker.patch_multimodal_merge import (
            _merge_multimodal_embeddings,
        )

        inputs_embeds = torch.zeros(5, 4, dtype=torch.float32)
        # Mask has 3 True positions but we only supply 2 embeddings.
        is_multimodal = torch.tensor(
            [True, False, True, False, True], dtype=torch.bool
        )
        mm_embeddings = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
            dtype=torch.float32,
        )

        with self.assertRaises(ValueError):
            _merge_multimodal_embeddings(
                inputs_embeds, [mm_embeddings], is_multimodal
            )
