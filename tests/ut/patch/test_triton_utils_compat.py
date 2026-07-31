# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest

from vllm_ascend.patch.triton_utils_compat import _UnavailableGluonModule
from vllm_ascend.version import vllm_version_is


@pytest.mark.skipif(vllm_version_is("0.26.0"), reason="the Gluon shim is only installed for target main")
def test_unavailable_gluon_module_allows_attribute_probes():
    module = _UnavailableGluonModule("triton.experimental.gluon")

    assert not hasattr(module, "__file__")
    assert not hasattr(module, "torch")

    with pytest.raises(
        AttributeError,
        match="has no attribute 'jit'",
    ):
        _ = module.jit
