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
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.compilation.backends import VllmBackend

from tests.ut.base import TestBase
from vllm_ascend.patch.worker import patch_vpp_compile as vpp_compile


class FakeModel:
    """Minimal stand-in for a ``@support_torch_compile`` model."""

    code_object = object()

    def __init__(self):
        self.do_not_compile = False
        self.vllm_config = object()
        self._compile_prefix = "backbone"
        self._is_encoder = False
        self.compiled = False
        self._compiled_callable = "pristine"
        self._compiled_bytecode = None
        self.was_aot_compile_fn_loaded_from_disk = False
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1

    def original_code_object(self):
        return self.code_object


class TestBindCompileStage(TestBase):
    def setUp(self):
        super().setUp()
        self.model = FakeModel()
        self.reinitialized = []

        def _record(model):
            # The stage must already be active when the wrapper is rebuilt:
            # the backend prefix and the AOT hash both read it.
            self.reinitialized.append(vpp_compile.current_vpp_compile_stage())
            model.cleanup_calls += 1

        self.reinit_patch = patch.object(vpp_compile, "_reinitialize_wrapper", side_effect=_record)
        self.reinit_patch.start()
        self.addCleanup(self.reinit_patch.stop)
        self.addCleanup(setattr, vpp_compile, "_CURRENT_VPP_STAGE", None)

    def test_disabled_model_is_left_alone(self):
        self.model.do_not_compile = True
        self.assertIsNone(vpp_compile.bind_vpp_compile_stage(self.model, 0))
        self.assertIsNone(vpp_compile.current_vpp_compile_stage())
        self.assertEqual(self.reinitialized, [])

    def test_pristine_wrapper_is_adopted_for_first_stage(self):
        binding = vpp_compile.bind_vpp_compile_stage(self.model, 0)
        self.assertIsNotNone(binding)
        self.assertEqual(self.reinitialized, [])
        self.assertEqual(vpp_compile.current_vpp_compile_stage(), 0)

        vpp_compile.capture_vpp_compile_stage(binding)
        self.assertIsNone(vpp_compile.current_vpp_compile_stage())
        self.assertIn(0, self.model._vpp_compile_artifacts)
        self.assertEqual(self.model._vpp_compile_artifacts[0]["_compiled_callable"], "pristine")

    def test_next_stage_gets_its_own_wrapper(self):
        vpp_compile.capture_vpp_compile_stage(vpp_compile.bind_vpp_compile_stage(self.model, 0))

        binding = vpp_compile.bind_vpp_compile_stage(self.model, 1)
        self.assertEqual(self.reinitialized, [1])
        self.assertEqual(vpp_compile.current_vpp_compile_stage(), 1)
        vpp_compile.capture_vpp_compile_stage(binding)

    def test_artifact_is_installed_without_recompiling(self):
        binding = vpp_compile.bind_vpp_compile_stage(self.model, 0)
        self.model.aot_compiled_fn = "stage0-fn"
        self.model.compiled = True
        vpp_compile.capture_vpp_compile_stage(binding)

        binding = vpp_compile.bind_vpp_compile_stage(self.model, 1)
        self.assertEqual(self.reinitialized, [1])
        self.model.aot_compiled_fn = "stage1-fn"
        vpp_compile.capture_vpp_compile_stage(binding)

        # Coming back to stage 0 restores its state, so its compiled callable
        # is dispatched instead of the one stage 1 left behind.
        binding = vpp_compile.bind_vpp_compile_stage(self.model, 0)
        self.assertEqual(self.reinitialized, [1])
        self.assertEqual(self.model.aot_compiled_fn, "stage0-fn")
        self.assertTrue(self.model.compiled)
        vpp_compile.capture_vpp_compile_stage(binding)

    def test_compile_during_graph_capture_is_rejected(self):
        # Reusing a stage's artifacts needs no compilation and stays legal
        # even inside a capture; building a new stage's does not.
        with (
            patch.object(torch.npu, "is_current_stream_capturing", return_value=True),
            self.assertRaises(RuntimeError),
        ):
            vpp_compile.bind_vpp_compile_stage(self.model, 1)
        # The failed bind must not leave the stage behind for later compiles.
        self.assertIsNone(vpp_compile.current_vpp_compile_stage())

    def test_context_manager_captures_on_failure(self):
        with self.assertRaises(ValueError), vpp_compile.vpp_compile_stage(self.model, 0):
            raise ValueError("forward failed")
        self.assertIn(0, self.model._vpp_compile_artifacts)
        self.assertIsNone(vpp_compile.current_vpp_compile_stage())


class TestReinitializeWrapper(TestBase):
    def setUp(self):
        super().setUp()
        self.model = FakeModel()
        self.model.aot_compiled_fn = "stale"
        self.model.compiled = True
        self.model.was_aot_compile_fn_loaded_from_disk = True
        self.addCleanup(setattr, vpp_compile, "_CURRENT_VPP_STAGE", None)
        vpp_compile._CURRENT_VPP_STAGE = 1

    def test_state_is_cleared_and_wrapper_rebuilt(self):
        calls = {}

        def fake_init(model, **kwargs):
            calls["model"] = model
            calls["kwargs"] = kwargs

        with (
            patch.object(torch._dynamo.eval_frame, "remove_from_cache") as remove,
            patch(
                "vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper.__init__",
                fake_init,
            ),
        ):
            vpp_compile._reinitialize_wrapper(self.model)

        self.assertEqual(self.model.cleanup_calls, 1)
        self.assertIsNone(self.model.aot_compiled_fn)
        self.assertIsNone(self.model._compiled_callable)
        self.assertIsNone(self.model._compiled_bytecode)
        self.assertFalse(self.model.compiled)
        self.assertFalse(self.model.was_aot_compile_fn_loaded_from_disk)
        remove.assert_called_once_with(FakeModel.code_object)
        self.assertIs(calls["model"], self.model)
        self.assertEqual(calls["kwargs"]["compile_prefix"], "backbone")
        self.assertFalse(calls["kwargs"]["is_encoder"])


class TestCacheKeys(TestBase):
    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, vpp_compile, "_CURRENT_VPP_STAGE", None)
        vpp_compile._CURRENT_VPP_STAGE = None

    def _factors(self):
        from vllm.compilation import caching

        config = SimpleNamespace(compute_hash=lambda: "config-hash")
        with (
            patch.object(caching.envs, "compile_factors", return_value=("e",)),
            patch.object(caching.envs, "VLLM_USE_MEGA_AOT_ARTIFACT", False),
        ):
            return caching.aot_compile_hash_factors(config)

    def test_hash_factors_are_untouched_without_vpp(self):
        from vllm.compilation import caching

        self.assertIsNot(vpp_compile._ORIGINAL_AOT_HASH_FACTORS, None)
        self.assertIsNot(caching.aot_compile_hash_factors, vpp_compile._ORIGINAL_AOT_HASH_FACTORS)

        base = self._factors()
        self.assertEqual(base, self._factors())
        self.assertFalse(any(f.startswith("vpp_stage=") for f in base))

    def test_hash_factors_carry_the_active_stage(self):
        base = self._factors()
        vpp_compile._CURRENT_VPP_STAGE = 3
        self.assertEqual(self._factors(), [*base, "vpp_stage=3"])

    def test_backend_prefix_gets_one_stage_suffix(self):
        self.assertIsNot(vpp_compile._ORIGINAL_BACKEND_INIT, None)
        self.assertIsNot(VllmBackend.__init__, vpp_compile._ORIGINAL_BACKEND_INIT)
        self.assertEqual(vpp_compile._stage_prefixed_prefix("backbone", 0), "backbone__vp0")
        self.assertEqual(vpp_compile._stage_prefixed_prefix("backbone__vp0", 0), "backbone__vp0")
        self.assertEqual(vpp_compile._stage_prefixed_prefix("layers.0", 1), "layers.0__vp1")
