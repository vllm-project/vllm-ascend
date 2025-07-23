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

from typing import List, Optional
from unittest.mock import MagicMock, patch

import torch
from torch.library import Library

from tests.ut.base import TestBase
from vllm_ascend.patch.worker.patch_common.patch_utils import \
    ascend_direct_register_custom_op


class TestPatchUtils(TestBase):

    def setUp(self):
        super().setUp()

        self.mock_op_func = MagicMock()
        self.mock_op_func.__annotations__ = {
            'param1': list[int],
            'param2': Optional[list[int]],
            'param3': str
        }

        self.mock_fake_impl = MagicMock()
        self.mock_lib = MagicMock(spec=Library)

        self.op_name = "test_op"
        self.mutates_args = ["arg1"]
        self.dispatch_key = "NPU"
        self.tags = (torch.Tag.pt2_compliant_tag, )

        self.patch_infer_schema = patch(
            'vllm_ascend.patch.worker.patch_common.patch_utils.torch.library.infer_schema'
        )
        self.patch_vllm_lib = patch(
            'vllm_ascend.patch.worker.patch_common.patch_utils.vllm_lib')

        self.mock_infer_schema = self.patch_infer_schema.start()
        self.mock_vllm_lib = self.patch_vllm_lib.start()

        self.addCleanup(self.patch_infer_schema.stop)
        self.addCleanup(self.patch_vllm_lib.stop)

    def test_utils_patched(self):
        from vllm import utils

        self.assertIs(utils.direct_register_custom_op,
                      ascend_direct_register_custom_op)

    def test_register_with_default_lib(self):
        self.mock_infer_schema.return_value = "(Tensor self) -> Tensor"

        ascend_direct_register_custom_op(op_name=self.op_name,
                                         op_func=self.mock_op_func,
                                         mutates_args=self.mutates_args,
                                         fake_impl=self.mock_fake_impl,
                                         dispatch_key=self.dispatch_key,
                                         tags=self.tags)

        self.assertEqual(self.mock_op_func.__annotations__['param1'],
                         List[int])
        self.assertEqual(self.mock_op_func.__annotations__['param2'],
                         Optional[List[int]])
        self.assertEqual(self.mock_op_func.__annotations__['param3'], str)

        self.mock_infer_schema.assert_called_once_with(
            self.mock_op_func, mutates_args=self.mutates_args)

        self.mock_vllm_lib.define.assert_called_once_with(
            f"{self.op_name}(Tensor self) -> Tensor", tags=self.tags)
        self.mock_vllm_lib.impl.assert_called_once_with(
            self.op_name, self.mock_op_func, dispatch_key=self.dispatch_key)
        self.mock_vllm_lib._register_fake.assert_called_once_with(
            self.op_name, self.mock_fake_impl)

    def test_register_with_custom_lib(self):
        self.mock_infer_schema.return_value = "(Tensor a, Tensor b) -> Tensor"

        ascend_direct_register_custom_op(op_name=self.op_name,
                                         op_func=self.mock_op_func,
                                         mutates_args=self.mutates_args,
                                         target_lib=self.mock_lib)

        self.mock_lib.define.assert_called_once_with(
            f"{self.op_name}(Tensor a, Tensor b) -> Tensor", tags=())
        self.mock_lib.impl.assert_called_once_with(self.op_name,
                                                   self.mock_op_func,
                                                   dispatch_key="CUDA")
        self.mock_lib._register_fake.assert_not_called()
