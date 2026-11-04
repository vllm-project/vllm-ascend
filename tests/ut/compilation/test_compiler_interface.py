# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_ascend.compilation.compiler_interface import _initialize_static_kernel_runtime


def test_static_kernel_excludes_the_registered_gbsa_operator(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    blacklist_setter = Mock()
    with patch("vllm_ascend.compilation.compiler_interface._set_static_kernel_blacklist", blacklist_setter):
        _initialize_static_kernel_runtime(SimpleNamespace())
    blacklist_setter.assert_called_once()
    blacklist = blacklist_setter.call_args.args[0]
    assert "GenericBlockSparseAttention" in blacklist
    assert "GenericBlockSparseAttn" not in blacklist
    assert any(op in "static_kernel_GenericBlockSparseAttention_001.json" for op in blacklist)
    assert not any(op in "static_kernel_MatMulV2_001.json" for op in blacklist)
