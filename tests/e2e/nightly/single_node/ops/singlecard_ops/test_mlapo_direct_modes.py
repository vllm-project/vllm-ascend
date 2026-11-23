import gc
import socket

import pytest
import torch
import torch_npu

from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import scale_from_float_to_int64
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module")
def mapped_offload():
    from memfabric_hybrid import offload  # type: ignore[import-untyped]

    assert enable_custom_op()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cfg = offload.OffloadConfig()
    cfg.store_url = f"tcp://127.0.0.1:{port}"
    cfg.device_id = torch.npu.current_device()
    cfg.reserve_size = cfg.alloc_size = 1 << 30
    cfg.world_size, cfg.rank_id, cfg.scene = 1, 0, offload.Scene.SHARED
    assert offload.initialize(cfg) == 0
    try:
        yield offload
    finally:
        torch.npu.synchronize()
        gc.collect()
        offload.uninitialize()


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Exercises A3 mapped CPU writeback")
@pytest.mark.parametrize(
    "dtype,quant_mode",
    [
        (torch.bfloat16, "per_tensor_quant_asymm"),
        (torch.float16, "per_tensor_quant_asymm"),
        (torch.bfloat16, "per_token_quant_symm"),
    ],
)
@torch.inference_mode()
def test_mlapo_direct_matches_original_and_replays_writeback(dtype, quant_mode, mapped_offload, monkeypatch):
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    torch.manual_seed(0)
    tokens, heads, hidden_size, q_rank, kv_rank, rope_dim, nope_dim, block_size = 6, 2, 7168, 1536, 512, 64, 128, 128
    dims = (q_rank + kv_rank + rope_dim, heads * (nope_dim + rope_dim))
    hidden = torch.randn(tokens, hidden_size, dtype=dtype, device="npu") * 0.1
    wqkv, wq = (
        torch_npu.npu_format_cast(torch.randint(-3, 4, (1, width // 32, out, 32), dtype=torch.int8, device="npu"), 29)
        for width, out in zip((hidden_size, q_rank), dims)
    )
    descales = tuple(torch.full((dim,), 0.005, dtype=torch.float32, device="npu") for dim in dims)
    if dtype == torch.float16:
        descales = tuple(scale_from_float_to_int64(value) for value in descales)
    cos = torch.rand(tokens, rope_dim, dtype=dtype, device="npu")
    sin = torch.rand_like(cos)
    inputs = (
        hidden,
        wqkv,
        descales[0],
        torch.ones(q_rank, dtype=dtype, device="npu"),
        torch.full((q_rank,), 0.1, dtype=dtype, device="npu"),
        wq,
        descales[1],
        torch.ones(kv_rank, dtype=dtype, device="npu"),
        cos,
        sin,
        torch.randn(heads, nope_dim, kv_rank, dtype=dtype, device="npu") * 0.01,
    )
    quant = dict(
        quant_scale0=torch.full((1,), 0.25, dtype=dtype, device="npu"),
        quant_offset0=torch.ones(1, dtype=torch.int8, device="npu"),
        bias0=torch.ones(dims[0], dtype=torch.int32, device="npu"),
        quant_scale1=torch.full((1,), 0.25, dtype=dtype, device="npu"),
        quant_offset1=torch.ones(1, dtype=torch.int8, device="npu"),
        bias1=torch.ones(dims[1], dtype=torch.int32, device="npu"),
        ctkv_scale=torch.ones(1, dtype=dtype, device="npu"),
        q_nope_scale=torch.ones(heads, dtype=dtype, device="npu"),
        quant_mode=quant_mode,
    )
    q_ref = tuple(
        torch.full(shape, float("nan"), dtype=dtype, device="npu")
        for shape in ((tokens, heads, kv_rank), (tokens, heads, rope_dim), (tokens, q_rank))
    )
    caches = tuple(
        torch.full((1, block_size, width), float("nan"), dtype=dtype, device="npu") for width in (kv_rank, rope_dim)
    )
    torch.ops._C_ascend.mla_preprocess(
        *inputs,
        *caches,
        torch.arange(tokens, dtype=torch.int32, device="npu"),
        **quant,
        cache_mode="krope_ctkv",
        enable_inner_out=True,
        q_out0=q_ref[0],
        q_out1=q_ref[1],
        inner_out=q_ref[2],
        kv_cache_out0=caches[0],
        kv_cache_out1=caches[1],
    )
    reference = tuple(value.cpu() for value in (*q_ref, caches[0][0, :tokens], caches[1][0, :tokens]))
    outputs = tuple(torch.full_like(value, float("nan"), device="npu") for value in reference)
    pools = tuple(
        mapped_offload.empty([block_size, width], dtype=dtype, pin_memory=True) for width in (kv_rank, rope_dim)
    )
    slots = torch.tensor([1, 4, -1, 7, 10, block_size], dtype=torch.int32, device="npu")

    def launch(**writeback):
        torch.ops._C_ascend.mla_preprocess_with_kv_offload(
            *inputs,
            **quant,
            q_out0=outputs[0],
            q_out1=outputs[1],
            inner_out=outputs[2],
            current_k=outputs[3],
            current_rope=outputs[4],
            **writeback,
        )

    launch()
    for actual, expected in zip(outputs, reference):
        assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
        assert torch.isfinite(expected).all()

    graph = UpdatableGraph()
    for pool in pools:
        pool.fill_(-17)
    with torch.npu.graph(graph, capture_error_mode="global"):
        launch(
            writeback_slots=slots,
            writeback_k_base=pools[0].data_ptr(),
            writeback_rope_base=pools[1].data_ptr(),
            writeback_capacity=block_size,
        )

    torch.npu.synchronize()
    slot_ids = [2, 5, -1, 8, 11, block_size]
    slots.copy_(torch.tensor(slot_ids, dtype=torch.int32, device="npu"))
    for pool in pools:
        pool.fill_(-17)
    for output in outputs:
        output.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()

    for actual, expected in zip(outputs, reference):
        assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
    for pool, current in zip(pools, reference[3:]):
        expected_pool = torch.full_like(pool, -17)
        for row, slot in enumerate(slot_ids):
            if 0 <= slot < block_size:
                expected_pool[slot] = current[row]
        assert torch.equal(pool.view(torch.int16), expected_pool.view(torch.int16))
