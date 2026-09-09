# SPDX-License-Identifier: Apache-2.0
"""Real CPU storage tests; also runnable with --confcutdir=tests/ut/worker."""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load the complete torch-only module without the plugin package initializer,
# which imports vLLM. This suite intentionally also works in a torch-only env.
_spec = importlib.util.spec_from_file_location(
    "sfa_kv_layout", Path(__file__).parents[3] / "vllm_ascend/worker/sfa_kv_layout.py"
)
_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_layout)
get_sfa_kv_parent = _layout.get_sfa_kv_parent
split_sfa_kv_parent = _layout.split_sfa_kv_parent


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset", [0, 32])
@pytest.mark.parametrize("shape", [(3, 4, 1, 12), (6, 2, 1, 12)])
def test_parent_roundtrip_and_cross_page_writes(dtype, offset, shape):
    size = 3 * 4 * 12 * 2
    backing = torch.zeros(size + offset + 32, dtype=torch.int8)
    raw = backing[offset : offset + size]
    k, r = split_sfa_kv_parent(raw, dtype=dtype, shape=shape, nope_dim=8)
    parent = get_sfa_kv_parent(k, r)
    assert parent.shape == shape
    assert parent.data_ptr() == raw.data_ptr()
    assert k.untyped_storage().data_ptr() == r.untyped_storage().data_ptr()
    assert r.storage_offset() - k.storage_offset() == 8
    assert k.stride() == r.stride() == (shape[1] * 12, 12, 12, 1)
    for block, token in [(0, 0), (0, 1), (0, shape[1] - 1), (1, 0), (shape[0] - 1, shape[1] - 1)]:
        k[block, token, 0] = 7
        r[block, token, 0] = 9
        assert torch.all(parent[block, token, 0, :8] == 7)
        assert torch.all(parent[block, token, 0, 8:] == 9)
    assert parent[2, 0].count_nonzero() == 0
    assert backing[:offset].count_nonzero() == 0
    assert backing[offset + size :].count_nonzero() == 0
    parent[1, 0, 0, 0] = 11
    assert k[1, 0, 0, 0] == 11


@pytest.mark.parametrize(
    "kind",
    ["dtype", "rank", "strided", "short", "padding", "unaligned", "heads", "width", "zero", "shape_rank", "quantized"],
)
def test_split_rejects_invalid_raw_or_geometry(kind):
    raw = torch.zeros(288, dtype=torch.int8)
    shape, width, dtype = (3, 4, 1, 12), 8, torch.float16
    if kind == "dtype":
        raw = raw.to(torch.uint8)
    if kind == "rank":
        raw = raw.view(3, 96)
    if kind == "strided":
        raw = torch.zeros(576, dtype=torch.int8)[::2]
    if kind == "short":
        raw = raw[:-2]
    if kind == "padding":
        raw = torch.zeros(300, dtype=torch.int8)
    if kind == "unaligned":
        raw = torch.zeros(289, dtype=torch.int8)[1:]
    if kind == "heads":
        shape = (3, 2, 2, 12)
    if kind == "width":
        width = 12
    if kind == "zero":
        shape = (0, 4, 1, 12)
    if kind == "shape_rank":
        shape = (3, 4, 12)
    if kind == "quantized":
        dtype = torch.int8
    with pytest.raises(ValueError):
        split_sfa_kv_parent(raw, dtype=dtype, shape=shape, nope_dim=width)


@pytest.mark.parametrize("kind", ["separate", "offset", "stride", "rank", "shape", "dtype", "heads", "empty", "padded"])
def test_reconstruction_rejects_non_parent_views(kind):
    p = torch.zeros(3, 4, 1, 12, dtype=torch.float16)
    k, r = p[..., :8], p[..., 8:]
    if kind == "separate":
        r = r.clone()
    if kind == "offset":
        r = p[..., 7:11]
    if kind == "stride":
        r = r.transpose(0, 1)
    if kind == "rank":
        k, r = k.squeeze(2), r.squeeze(2)
    if kind == "shape":
        r = r[:2]
    if kind == "dtype":
        r = r.to(torch.bfloat16)
    if kind == "heads":
        p = torch.zeros(3, 2, 2, 12, dtype=torch.float16)
        k, r = p[..., :8], p[..., 8:]
    if kind == "empty":
        k, r = k[:0], r[:0]
    if kind == "padded":
        p = torch.zeros(3, 4, 1, 16, dtype=torch.float16)
        k, r = p[..., :8], p[..., 8:12]
    with pytest.raises(ValueError):
        get_sfa_kv_parent(k, r)


@pytest.mark.parametrize(
    "connector, module, expected",
    [
        (None, None, True),
        ("SfaRemoteD2HConnector", None, True),
        ("SfaRemoteD2HConnector", "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.connector", True),
        ("SfaRemoteD2HConnector", "custom.connector", False),
        ("MultiConnector", None, False),
        ("AscendStoreConnector", None, False),
        ("MooncakeConnector", None, False),
        ("MemcacheConnector", None, False),
    ],
)
def test_transfer_policy_keeps_unadapted_connectors_on_legacy_layout(connector, module, expected):
    from types import SimpleNamespace

    config = None if connector is None else SimpleNamespace(kv_connector=connector, kv_connector_module_path=module)
    assert _layout.sfa_kv_parent_supported_for_transfer(config) is expected
