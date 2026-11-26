# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, call, patch

from vllm_ascend.utils import no_super_kernel_for_multistream, super_kernel_scope


class TestSuperKernelMultistreamScope(unittest.TestCase):
    def test_only_enabled_overlap_inside_sk_capture_is_excluded(self):
        npu = MagicMock()
        with patch("vllm_ascend.utils.torch.npu", npu):
            with no_super_kernel_for_multistream(True):
                pass
            with super_kernel_scope("full_model", False), no_super_kernel_for_multistream(True):
                pass
            with super_kernel_scope("full_model", True):
                with no_super_kernel_for_multistream(False):
                    pass
                with no_super_kernel_for_multistream(True):
                    pass

        self.assertEqual(npu.super_kernel_scope_begin.call_args_list, [call("full_model"), call(None)])
        self.assertEqual(npu.super_kernel_scope_end.call_args_list, [call(None), call("full_model")])

    def test_scope_state_is_restored_after_exception(self):
        npu = MagicMock()
        with patch("vllm_ascend.utils.torch.npu", npu):
            with self.assertRaisesRegex(ValueError, "failure"):
                with super_kernel_scope("full_model", True), no_super_kernel_for_multistream(True):
                    raise ValueError("failure")
            with no_super_kernel_for_multistream(True):
                pass

        self.assertEqual(npu.super_kernel_scope_begin.call_args_list, [call("full_model"), call(None)])
        self.assertEqual(npu.super_kernel_scope_end.call_args_list, [call(None), call("full_model")])
