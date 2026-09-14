# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only tests loading the host selector without the NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_selector():
    root = Path(__file__).resolve().parents[4]
    source = root / "vllm_ascend/_310p/ops/fla/gdn_310.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prefill_cu_seqlens_310"
    )
    namespace = {"torch": torch, "GDNAttentionMetadata": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("boundaries", [(0, 1280), (0, 768), (0, 2048), (0, 0, 65, 65, 129)])
def test_host_boundaries_match_device_values_and_preserve_empty_segments(boundaries):
    device_cu = torch.tensor(boundaries, dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=0,
        non_spec_query_start_loc=device_cu,
        non_spec_prefill_metadata=SimpleNamespace(chunk=SimpleNamespace(cu_seqlens_host=boundaries)),
    )
    selected = _load_selector()(metadata)
    assert selected.device.type == "cpu"
    assert selected.dtype == torch.int64
    assert selected.tolist() == device_cu.tolist()
    assert selected is not device_cu
    selected.zero_()
    assert metadata.non_spec_prefill_metadata.chunk.cu_seqlens_host == boundaries
    assert device_cu.tolist() == list(boundaries)


@pytest.mark.parametrize("decodes,host", [(1, (0, 2048)), (0, None), (0, (0, 1, 2048))])
def test_mixed_or_missing_host_metadata_keeps_existing_device_path(decodes, host):
    device_cu = torch.tensor([0, 2048])
    metadata = SimpleNamespace(
        num_decodes=decodes,
        non_spec_query_start_loc=device_cu,
        non_spec_prefill_metadata=SimpleNamespace(chunk=SimpleNamespace(cu_seqlens_host=host)),
    )
    assert _load_selector()(metadata) is device_cu


def test_older_metadata_without_prefill_attachment_keeps_existing_path():
    device_cu = torch.tensor([0, 2048])
    metadata = SimpleNamespace(num_decodes=0, non_spec_query_start_loc=device_cu)
    assert _load_selector()(metadata) is device_cu


def test_fast_path_does_not_read_or_copy_device_values():
    class DeviceBoundaries:
        def numel(self):
            return 2

        def __getattr__(self, name):
            raise AssertionError(f"Unexpected device access: {name}")

    metadata = SimpleNamespace(
        num_decodes=0,
        non_spec_query_start_loc=DeviceBoundaries(),
        non_spec_prefill_metadata=SimpleNamespace(chunk=SimpleNamespace(cu_seqlens_host=(0, 2048))),
    )
    assert _load_selector()(metadata).tolist() == [0, 2048]
