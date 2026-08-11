# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Helpers for loading ACLNN custom operators in real-device sampling UTs."""

import pytest
import torch

from vllm_ascend.utils import enable_categorical_sample_op


def require_categorical_sampling_operator() -> None:
    """Load the categorical sampler through the platform's supported path.

    A2/A3 use the general lazy custom-op initializer. A5 deliberately keeps
    that initializer disabled because its complete direct-kernel set is not
    supported. Its ACLNN operators are nevertheless supported through the
    packaged custom OPP and native Torch registration module, which mirrors
    the established A5 operator-test loading path.
    """
    if not enable_categorical_sample_op():
        pytest.fail("npu_categorical_sample requires the categorical custom operator")

    if not hasattr(torch.ops._C_ascend, "npu_categorical_sample"):
        pytest.fail("the installed extension does not register npu_categorical_sample")
