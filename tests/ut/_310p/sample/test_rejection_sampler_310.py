#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import vllm_ascend.sample.rejection_sampler as rejection_sampler_module
from tests.ut.base import TestBase
from vllm_ascend._310p.sample.rejection_sampler import _force_pytorch_rejection_path


class TestForcePytorchRejectionPath(TestBase):
    def test_disables_triton_and_binds_recovered_then_restores(self):
        orig_triton = rejection_sampler_module.HAS_TRITON
        orig_recovered = rejection_sampler_module.sample_recovered_tokens

        def sentinel(*args, **kwargs):
            return None

        with _force_pytorch_rejection_path(sentinel):
            # 310P has no working Triton; the base sampler must take PyTorch paths.
            self.assertFalse(rejection_sampler_module.HAS_TRITON)
            self.assertIs(rejection_sampler_module.sample_recovered_tokens, sentinel)

        self.assertEqual(rejection_sampler_module.HAS_TRITON, orig_triton)
        self.assertIs(rejection_sampler_module.sample_recovered_tokens, orig_recovered)

    def test_restores_on_exception(self):
        orig_triton = rejection_sampler_module.HAS_TRITON
        orig_recovered = rejection_sampler_module.sample_recovered_tokens

        with self.assertRaises(RuntimeError):
            with _force_pytorch_rejection_path(lambda *a, **k: None):
                raise RuntimeError("boom")

        self.assertEqual(rejection_sampler_module.HAS_TRITON, orig_triton)
        self.assertIs(rejection_sampler_module.sample_recovered_tokens, orig_recovered)
