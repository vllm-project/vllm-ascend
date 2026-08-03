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
# This file is a part of the vllm-ascend project.
#
"""Signature-drift guard for the Ascend mamba postprocess kernel.

``patch_mamba_utils`` swaps the Ascend kernel in for
``vllm.v1.worker.mamba_utils.postprocess_mamba_fused_kernel`` but leaves the
launcher upstream, so the two parameter lists must stay identical. When upstream
adds a parameter and the Ascend kernel does not follow, every launch raises
``TypeError`` and hybrid (Gated-DeltaNet) models such as Qwen3.5/Qwen3.6 fail to
serve.

The upstream names are read from the vLLM source file rather than from the
module attribute: by the time this test runs the patch has already replaced the
attribute, so comparing objects would be vacuous.
"""

from __future__ import annotations

import ast
import inspect

from vllm.v1.worker import mamba_utils as upstream_mamba_utils

from vllm_ascend.ops.triton.mamba.postprocess import postprocess_mamba_fused_kernel

_KERNEL_NAME = "postprocess_mamba_fused_kernel"


def _upstream_param_names() -> list[str]:
    """Parameter names of the upstream kernel, parsed from its source file."""
    source_file = inspect.getsourcefile(upstream_mamba_utils)
    assert source_file is not None, "cannot locate vllm.v1.worker.mamba_utils source"
    with open(source_file, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _KERNEL_NAME:
            args = node.args
            return [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    raise AssertionError(f"{_KERNEL_NAME} not found in {source_file}")


def _ascend_param_names() -> list[str]:
    # triton.jit wraps the function; the original is kept on `.fn`.
    fn = getattr(postprocess_mamba_fused_kernel, "fn", postprocess_mamba_fused_kernel)
    return list(inspect.signature(fn).parameters)


def test_kernel_signature_matches_upstream():
    assert _ascend_param_names() == _upstream_param_names(), (
        "Ascend postprocess_mamba_fused_kernel drifted from the upstream vLLM "
        "kernel it replaces; the upstream launcher passes arguments both "
        "positionally and by keyword, so the parameter lists must match exactly."
    )
