# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test dispatcher contracts used when tracing without the DSL compiler.

Numerical and device graph coverage lives in the singlecard DSV4.1 DSL suite.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.dsv41_a5 import dsl as ops


@pytest.mark.parametrize(
    "name", ["quant_lightning_indexer", "quant_sparse_lightning_indexer", "mixed_quant_sparse_flash_mla"]
)
@pytest.mark.parametrize(
    "package_version,cores,missing,local",
    [
        ("0.1.0", 28, None, True),
        ("0.1.0", 32, None, False),
        ("0.1.0", 36, None, False),
        ("0.2.0", 28, None, False),
        ("0.2.0", 28, "module", True),
        ("0.2.0", 28, "api", True),
    ],
)
def test_indexer_selects_compute_and_metadata_together(monkeypatch, name, package_version, cores, missing, local):
    ops._get_dsl_ops.cache_clear()
    if name == "mixed_quant_sparse_flash_mla" and missing is None:
        local = False
    external_pair = (lambda: None, lambda: None)
    local_pair = (lambda: None, lambda: None)
    imports = []

    def import_module(module):
        imports.append(module)
        is_metadata = module.endswith(("_metadata_dsl", "_metadata"))
        if module.startswith("ops."):
            if missing == "module" and is_metadata:
                raise ModuleNotFoundError(name=module)
            if missing == "api" and is_metadata:
                return SimpleNamespace()
            pair = external_pair
        else:
            pair = local_pair
        return SimpleNamespace(**{f"{name}_metadata" if is_metadata else name: pair[int(is_metadata)]})

    monkeypatch.setattr(ops, "version", lambda _: package_version)
    monkeypatch.setattr(ops, "import_module", import_module)
    monkeypatch.setattr(torch.npu, "get_device_properties", lambda _: SimpleNamespace(cube_core_num=cores))
    try:
        result = ops._get_dsl_ops(name, torch.device("npu", 0))
        assert result == (local_pair if local else external_pair)
        count = len(imports)
        assert ops._get_dsl_ops(name, torch.device("npu", 0)) is result
        assert len(imports) == count
        if local and package_version == "0.1.0" and cores < 32:
            assert all(module.startswith("vllm_ascend.ops.pythondsl.") for module in imports)
    finally:
        ops._get_dsl_ops.cache_clear()


def test_indexer_does_not_hide_broken_package_dependency(monkeypatch):
    ops._get_dsl_ops.cache_clear()
    monkeypatch.setattr(ops, "version", lambda _: "0.2.0")

    def broken_import(module):
        raise ModuleNotFoundError(name="cannbotdsl")

    monkeypatch.setattr(ops, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError) as error:
        ops._get_dsl_ops("quant_lightning_indexer", torch.device("npu", 0))
    assert error.value.name == "cannbotdsl"


def test_missing_operator_package_uses_local_pair(monkeypatch):
    ops._get_dsl_ops.cache_clear()
    compute, metadata = lambda: None, lambda: None

    def missing_version(_):
        raise ops.PackageNotFoundError

    def import_module(module):
        if module.startswith("ops."):
            raise ModuleNotFoundError(name="ops")
        if module.endswith("_metadata_dsl"):
            return SimpleNamespace(quant_lightning_indexer_metadata=metadata)
        return SimpleNamespace(quant_lightning_indexer=compute)

    monkeypatch.setattr(ops, "version", missing_version)
    monkeypatch.setattr(ops, "import_module", import_module)
    try:
        assert ops._get_dsl_ops("quant_lightning_indexer", torch.device("npu", 0)) == (compute, metadata)
    finally:
        ops._get_dsl_ops.cache_clear()


@pytest.mark.parametrize("return_value,candidate_blocks", [(False, -1), (True, 2048)])
def test_indexer_meta_pipeline(return_value, candidate_blocks):
    q = torch.empty((7, 32, 64), dtype=torch.uint8, device="meta")
    k = torch.empty((3, 128, 1, 72), dtype=torch.uint8, device="meta")
    w = torch.empty((7, 32), dtype=torch.float32, device="meta")
    scale = torch.empty((7, 32, 1), dtype=torch.float32, device="meta")
    k_scale = torch.empty((3, 128, 1), dtype=torch.float32, device="meta")
    indices, values, candidates, lengths = ops.quant_lightning_indexer(
        q,
        k,
        w,
        scale,
        k_scale,
        512,
        0,
        layout_k="PA_BBND",
        return_value=return_value,
        candidate_topk_blocks=candidate_blocks,
        candidate_block_size=8,
    )
    assert indices.shape == (7, 1, 512)
    assert indices.dtype == torch.int32
    assert values.shape == ((7, 1, 512) if return_value else (0,))
    assert values.dtype == torch.bfloat16
    assert all(t.device.type == "meta" for t in (indices, values, candidates, lengths))
    assert candidates.dtype == lengths.dtype == torch.int32
    if candidate_blocks < 0:
        assert candidates.shape == lengths.shape == (0,)
        return

    assert candidates.shape == (7, 1, 2048)
    assert lengths.shape == (7, 1)
    sparse_indices, sparse_values = ops.quant_sparse_lightning_indexer(
        q,
        k,
        w,
        scale,
        candidates,
        lengths,
        512,
        0,
        8,
        descale_k=k_scale,
        layout_k="PA_BBND",
        return_value=return_value,
    )
    assert sparse_indices.shape == indices.shape
    assert sparse_indices.dtype == indices.dtype
    assert sparse_values.shape == values.shape
    assert sparse_values.dtype == values.dtype
    assert sparse_indices.device.type == sparse_values.device.type == "meta"


@pytest.mark.parametrize(
    "name",
    [
        "quant_lightning_indexer_metadata",
        "quant_sparse_lightning_indexer_metadata",
        "mixed_quant_sparse_flash_mla_metadata",
    ],
)
def test_metadata_meta_abi(name):
    lengths = torch.empty((7, 1), dtype=torch.int32, device="meta")
    cu = torch.empty((3,), dtype=torch.int32, device="meta")
    if name == "mixed_quant_sparse_flash_mla_metadata":
        metadata = getattr(ops, name)(
            lengths,
            lengths,
            cu_seqlens_q=cu,
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=512,
            quant_mode=0,
        )
    else:
        kwargs = dict(cu_seqlens_q=cu, num_heads_q=32, num_heads_k=1, head_dim=128, topk=512)
        if name == "quant_sparse_lightning_indexer_metadata":
            kwargs.update(candidate_block_length=lengths, quant_mode=0, candidate_block_size=8)
        metadata = getattr(ops, name)(**kwargs)
    assert metadata.shape == (1024,)
    assert metadata.dtype == torch.int32
    assert metadata.device.type == "meta"


@pytest.mark.parametrize("return_lse", [False, True])
def test_attention_meta_outputs(return_lse):
    q = torch.empty((7, 64, 512), dtype=torch.bfloat16, device="meta")
    kv = torch.empty((3, 128, 1, 584), dtype=torch.uint8, device="meta")
    output, lse = ops.mixed_quant_sparse_flash_mla(
        q,
        ori_kv=kv,
        quant_mode=0,
        return_softmax_lse=return_lse,
    )
    assert output.shape == (7, 64, 512)
    assert output.dtype == torch.bfloat16
    assert lse.shape == ((1, 7, 64) if return_lse else (0,))
    assert lse.dtype == torch.float32
    assert output.device.type == lse.device.type == "meta"
