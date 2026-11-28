# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request NPU projection checks; prefill can contain multiple tokens.

Adapters switch serially. These checks do not accept the complete GLM model
or establish multi-request batch invariance.
"""
# Optional NPU availability must be checked before importing the NPU runtime.
# ruff: noqa: E402

from types import SimpleNamespace

import pytest
import torch
from torch import nn

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("GLM LoRA projection acceptance requires a real NPU", allow_module_level=True)

import cann_ops_transformer  # noqa: F401
from vllm.config import LoRAConfig
from vllm.lora.layers import LoRAMapping

from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.lora.utils import AscendColumnParallelLinearWithLoRA
from vllm_ascend.ops.cann_keypool import prepare_keypool_state
from vllm_ascend.ops.triton.glm5_next_cann_keypool import apply_keypool_lora, write_keypool_caches
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def initialize_npu_operators():
    torch.npu.set_device(0)
    assert enable_custom_op(), "GLM LoRA projection acceptance requires Ascend custom operators"


@torch.inference_mode()
def _keypool_step_metadata(start, count, device, capacity):
    pool, stop = 4, start + count
    complete = stop // pool - start // pool
    pooled_rows = count // pool + 1
    slots = torch.full((pooled_rows,), -1, dtype=torch.int64, device=device)
    slots[:complete] = torch.arange(start // pool, stop // pool, device=device)
    metadata = SimpleNamespace(
        write_metadata=None,
        compress_ratio=pool,
        positions=torch.arange(start, stop, device=device),
        cum_query_lens=torch.tensor([count], dtype=torch.int64, device=device),
        raw_seq_lens=torch.tensor([stop], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, count], dtype=torch.int32, device=device),
        start_pos=torch.tensor([start], dtype=torch.int32, device=device),
        keypool_start_offsets=torch.tensor([start % pool], dtype=torch.int32, device=device),
        pool_tail=torch.tensor([stop % pool], dtype=torch.int64, device=device),
        pooled_slot_mapping=slots,
        keypool_state_table=torch.tensor([[1, 2]], dtype=torch.int32, device=device),
    )
    offsets = torch.arange(pool, device=device)
    writes = torch.where(offsets < stop % pool, (stop - stop % pool + offsets) % capacity, -1)[None]
    sources = ((1 + int(start % pool + count >= pool)) * pool + offsets)[None]
    inputs = SimpleNamespace(
        history_indices=((start - start % pool + offsets) % capacity)[None],
        write_indices=writes,
        state_indices=sources,
        retained_indices=None,
        page_rows=capacity,
    )
    return metadata, SimpleNamespace(block_size=capacity, cann_inputs=inputs)


@pytest.mark.parametrize(
    "cache_dtype",
    [
        torch.bfloat16,
        pytest.param(
            torch.float8_e4m3fn,
            marks=pytest.mark.skipif(
                get_ascend_device_type() != AscendDeviceType.A5, reason="FP8 LI cache requires Ascend 950"
            ),
        ),
        torch.int8,
    ],
)
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("width,chunks", [(1024, (3, 1, 4, 1, 3)), (4096, (16384,))])
@torch.inference_mode()
def test_indexer_lora_survives_prefill_decode_pool_boundary(cache_dtype, zero, width, chunks):
    torch.npu.set_device(0)
    torch.manual_seed(55)
    device, dim, heads, rank = torch.device("npu:0"), 128, 32, 8
    total = sum(chunks)
    config = LoRAConfig(max_loras=1, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(total, 1, device, lora_config=config)
    x = torch.randn(total, width, device=device, dtype=torch.bfloat16) * 0.1
    weight = torch.randn(dim + heads, width, device=device, dtype=torch.bfloat16) * 0.1
    aa = tuple(torch.randn(1, 1, rank, width, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(2))
    bb = tuple(torch.randn(1, 1, size, rank, device=device, dtype=torch.bfloat16) * 0.1 for size in (dim, heads))
    if zero:
        for value in bb:
            value.zero_()
    projection = object.__new__(AscendColumnParallelLinearWithLoRA)
    nn.Module.__init__(projection)
    projection.base_layer = nn.Linear(width, dim + heads, bias=False, device=device, dtype=torch.bfloat16)
    projection.base_layer.weight.data.copy_(weight)
    projection.punica_wrapper = punica
    projection.lora_a_stacked, projection.lora_b_stacked = aa, bb
    projection.output_slices = (dim, heads)
    norm = nn.LayerNorm(dim, device=device, dtype=torch.float32)
    blocks = total // 4 + 1
    cache = torch.full((1, blocks, 1, dim), 7, dtype=cache_dtype, device=device)
    scale = (
        None
        if cache_dtype == torch.bfloat16
        else torch.full(
            (1, blocks, 1, 1),
            -7,
            device=device,
            dtype=torch.float32 if cache_dtype == torch.float8_e4m3fn else torch.float16,
        )
    )
    tail = torch.zeros(1, 4, 2 * dim, device=device, dtype=torch.float32)
    native_cache, native_tail = cache.clone(), tail.clone()
    native_scale = scale.clone() if scale is not None else None
    compress_gate = torch.randn(dim, width, device=device, dtype=torch.bfloat16) * 0.1
    owner = SimpleNamespace(
        wk_weights_proj=projection,
        head_dim=dim,
        n_head=heads,
        _head_weight_f32=weight[dim:].float(),
        k_norm=norm,
        index_kpool=4,
        topk_tokens=2048,
        index_kpool_compress_gate=compress_gate,
        index_kpool_compress_ape=torch.zeros(4, dim, device=device),
        topk_indices_buffer=None,
        _prepare_keypool_state=prepare_keypool_state,
        _key_pool=torch.ops.cann_ops_transformer.key_pool,
        _apply_keypool_lora=apply_keypool_lora,
        _write_keypool_caches=write_keypool_caches,
        _symmetric_int8=Glm5NextKPoolIndexerBackend._symmetric_int8,
    )
    corrected_pools = []

    def apply_delta(pooled, state, delta, gate, ape, metadata, tail_metadata):
        apply_keypool_lora(pooled, state, delta, gate, ape, metadata, tail_metadata)
        valid = metadata.pooled_slot_mapping >= 0
        corrected_pools.append(pooled[valid].cpu())

    owner._apply_keypool_lora = apply_delta
    keys, projected_weights, start = [], [], 0
    for count in chunks:
        stop = start + count
        punica.update_metadata(LoRAMapping((1,) * count, (1,), is_prefill=count > 1), [1], 2, 128)
        metadata, tail_metadata = _keypool_step_metadata(start, count, device, 4)
        delta, weights = Glm5NextKPoolIndexerBackend._project_lora(owner, x[start:stop])
        base_key = norm(torch.nn.functional.linear(x[start:stop], weight[:dim]).float())
        keys.append((base_key + delta).cpu().double())
        projected_weights.append(weights.cpu().double())
        if zero:
            assert torch.count_nonzero(delta) == 0
        Glm5NextKPoolIndexerBackend._forward_cann(
            owner,
            x[start:stop],
            None,
            None,
            cache,
            tail,
            metadata,
            tail_metadata,
            compute_topk=False,
            indexer_scale_cache=scale,
            key_delta=delta,
        )
        if zero:
            Glm5NextKPoolIndexerBackend._forward_cann(
                owner,
                x[start:stop],
                None,
                None,
                native_cache,
                native_tail,
                metadata,
                tail_metadata,
                compute_topk=False,
                indexer_scale_cache=native_scale,
            )
            torch.testing.assert_close(cache, native_cache, atol=0, rtol=0)
            torch.testing.assert_close(tail, native_tail, atol=0, rtol=0)
            if scale is not None:
                torch.testing.assert_close(scale, native_scale, atol=0, rtol=0)
        start = stop
    torch.npu.synchronize()
    x_cpu = x.cpu().double()
    base_keys = (x_cpu @ weight[:dim].cpu().double().T).bfloat16()
    delta_keys = ((x_cpu @ aa[0][0, 0].cpu().double().T) @ bb[0][0, 0].cpu().double().T).bfloat16()
    expected_keys = torch.nn.functional.layer_norm((base_keys + delta_keys).double(), (dim,), eps=norm.eps)
    torch.testing.assert_close(torch.cat(keys), expected_keys, atol=3e-3, rtol=3e-2)
    # Punica expands into a BF16 delta; the base head GEMM remains FP32.
    head_delta = ((x_cpu @ aa[1][0, 0].cpu().double().T) @ bb[1][0, 0].cpu().double().T).bfloat16().double()
    expected_weights = x_cpu @ weight[dim:].cpu().double().T + head_delta
    torch.testing.assert_close(torch.cat(projected_weights), expected_weights, atol=5e-4, rtol=3e-2)
    scores = (x_cpu @ compress_gate.cpu().double().T).bfloat16().double().view(-1, 4, dim)
    expected = (scores.softmax(1).bfloat16().double() * expected_keys.view(-1, 4, dim)).sum(1).bfloat16().double()
    actual = cache[0, : total // 4, 0].cpu().double()
    if scale is not None:
        actual *= scale[0, : total // 4, 0].cpu().double()
    # Keep the original floating layer tolerance; verify the codec separately.
    corrected = torch.cat(corrected_pools).double()
    torch.testing.assert_close(corrected, expected, atol=3e-3, rtol=3e-2)
    if cache_dtype == torch.int8:
        expected_scale = (corrected.float().abs().amax(-1) / 127).clamp_min(torch.finfo(torch.float16).tiny).half()
        encoded = (corrected.float() / expected_scale.float()[:, None]).round().clamp(-127, 127).to(torch.int8)
        torch.testing.assert_close(cache[0, : total // 4, 0].cpu(), encoded, atol=0, rtol=0)
        torch.testing.assert_close(scale[0, : total // 4, 0, 0].cpu(), expected_scale, atol=0, rtol=0)
    elif cache_dtype == torch.float8_e4m3fn:
        expected_scale = corrected.float().abs().amax(-1) / 448
        encoded = (corrected.float() / expected_scale[:, None]).to(cache_dtype).float()
        torch.testing.assert_close(cache[0, : total // 4, 0].cpu().float(), encoded, atol=0, rtol=0)
        torch.testing.assert_close(scale[0, : total // 4, 0, 0].cpu(), expected_scale, atol=1e-6, rtol=2e-3)
    else:
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-2)
    assert torch.equal(cache[0, total // 4 :].cpu().float(), torch.full((1, 1, dim), 7.0))


@torch.inference_mode()
def test_zero_indexer_head_preserves_rank_local_projection_rounding():
    torch.manual_seed(58)
    device = torch.device("npu:0")
    count, width, dim, heads, rank = 16384, 4096, 128, 32, 8
    config = LoRAConfig(max_loras=1, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(count, 1, device, lora_config=config)
    punica.update_metadata(LoRAMapping((1,) * count, (1,), is_prefill=True), [1], 2, 154880)
    projection = object.__new__(AscendColumnParallelLinearWithLoRA)
    nn.Module.__init__(projection)
    projection.base_layer = nn.Linear(width, dim + heads, bias=False, device=device, dtype=torch.bfloat16)
    projection.punica_wrapper = punica
    projection.lora_a_stacked = tuple(
        torch.randn(1, 1, rank, width, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(2)
    )
    projection.lora_b_stacked = tuple(
        torch.zeros(1, 1, size, rank, device=device, dtype=torch.bfloat16) for size in (dim, heads)
    )
    projection.output_slices = (dim, heads)
    owner = SimpleNamespace(
        wk_weights_proj=projection,
        head_dim=dim,
        n_head=heads,
        _head_weight_f32=projection.base_layer.weight[dim:].float(),
    )
    inputs = torch.randn(count, width, device=device, dtype=torch.bfloat16)
    indices = punica.token_lora_indices.clone()
    for start, stop in ((0, 4096), (15863, count)):
        keys, weights = Glm5NextKPoolIndexerBackend._project_lora(
            owner, inputs, project_keys=False, head_rows=(start, stop)
        )
        expected = torch.nn.functional.linear(inputs[start:stop].float(), owner._head_weight_f32)
        assert keys is None
        torch.testing.assert_close(weights, expected, atol=0, rtol=0)
        torch.testing.assert_close(punica.token_lora_indices, indices, atol=0, rtol=0)


@pytest.mark.parametrize("prefill", [False, True])
def test_embedding_expand_switches_adapters_between_single_requests(prefill):
    torch.manual_seed(56)
    device, rank, hidden, vocab = torch.device("npu:0"), 8, 32, 31
    config = LoRAConfig(max_loras=2, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(8, 8, device, lora_config=config)
    a = torch.randn(2, rank, vocab, device=device, dtype=torch.bfloat16) * 0.1
    b = torch.randn(2, 1, hidden, rank, device=device, dtype=torch.bfloat16) * 0.1
    token_ids = [0, 15, vocab - 1, 4] if prefill else [vocab - 1]
    count = len(token_ids)
    base = torch.randn(count, hidden, device=device, dtype=torch.bfloat16) * 0.1
    for adapter_id in (1, 0, 2, 1):
        punica.update_metadata(LoRAMapping((adapter_id,) * count, (adapter_id,), is_prefill=prefill), [1, 2], 3, vocab)
        slot = max(adapter_id - 1, 0)
        # Keep base-only inputs nonzero so a missing mask cannot pass accidentally.
        inputs = torch.stack([a[slot, :, token] for token in token_ids])
        output = base.clone()
        punica.add_lora_embedding(output, inputs, b)
        torch.npu.synchronize()
        expected = base.cpu().double()
        if adapter_id:
            expected += inputs.cpu().double() @ b[slot, 0].cpu().double().T
        torch.testing.assert_close(output.cpu().double(), expected, atol=6e-4, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_lm_head_delta_preserves_padding_when_single_requests_switch_adapters(dtype):
    torch.manual_seed(57)
    device, rank, width, vocab, padded = torch.device("npu:0"), 8, 64, 29, 32
    config = LoRAConfig(max_loras=2, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(8, 8, device, lora_config=config)
    x = torch.randn(1, width, device=device, dtype=torch.bfloat16) * 0.1
    a = torch.randn(2, 1, rank, width, device=device, dtype=torch.bfloat16) * 0.1
    b = torch.randn(2, 1, padded, rank, device=device, dtype=torch.bfloat16) * 0.1
    b[:, :, vocab:].zero_()
    base = torch.randn(1, padded, device=device, dtype=dtype) * 0.1
    for adapter_id in (1, 0, 2, 1):
        punica.update_metadata(LoRAMapping((adapter_id,), (adapter_id,), is_prefill=False), [1, 2], 3, vocab)
        output = base.clone()
        punica.add_lora_logits(output, x, a, b, 1.0)
        torch.npu.synchronize()
        expected = base.cpu().double()
        if adapter_id:
            slot = adapter_id - 1
            expected += (x.cpu().double() @ a[slot, 0].cpu().double().T) @ b[slot, 0].cpu().double().T
        torch.testing.assert_close(output.cpu().double(), expected, atol=6e-4, rtol=3e-2)
        torch.testing.assert_close(output[:, vocab:], base[:, vocab:], rtol=0, atol=0)


def test_fused_experts_keep_expert_selection_when_single_requests_switch_adapters():
    torch.manual_seed(58)
    device, rank, width, intermediate = torch.device("npu:0"), 8, 64, 32
    config = LoRAConfig(max_loras=2, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(8, 8, device, lora_config=config)
    # Three prefill tokens from one request may route to different experts.
    experts = [1, 0, 1]
    a = tuple(torch.randn(2, 2, rank, width, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(2))
    b = tuple(torch.randn(2, 2, intermediate, rank, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(2))
    x = torch.randn(3, width, device=device, dtype=torch.bfloat16) * 0.1
    base = torch.randn(3, intermediate * 2, device=device, dtype=torch.bfloat16) * 0.1
    for slot in (1, -1, 0, 1):
        output = base.clone()
        punica.add_lora_fused_moe(
            output,
            x,
            a,
            b,
            expert_ids=torch.tensor(experts, device=device),
            adapter_enabled=torch.tensor([1, 1], device=device, dtype=torch.int32),
            token_lora_mapping=torch.full((3,), slot, device=device),
        )
        torch.npu.synchronize()
        expected = base.cpu().double()
        if slot >= 0:
            for row, expert in enumerate(experts):
                for shard in range(2):
                    expected[row, shard * intermediate : (shard + 1) * intermediate] += (
                        x[row].cpu().double() @ a[shard][slot, expert].cpu().double().T
                    ) @ b[shard][slot, expert].cpu().double().T
        torch.testing.assert_close(output.cpu().double(), expected, atol=6e-4, rtol=3e-2)


class TupleLinear(nn.Linear):
    def forward(self, inputs):
        return super().forward(inputs), None


@torch.inference_mode()
def test_tp_query_slice_keeps_single_request_adapter_selection():
    torch.manual_seed(5321)
    device, rank, width, output_width = torch.device("npu:0"), 8, 64, 128
    config = LoRAConfig(max_loras=2, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(8, 8, device, lora_config=config)
    projection = object.__new__(AscendColumnParallelLinearWithLoRA)
    nn.Module.__init__(projection)
    projection.base_layer = TupleLinear(width, output_width, bias=False, device=device, dtype=torch.bfloat16)
    projection.punica_wrapper = punica
    a = torch.randn(2, 1, rank, width, device=device, dtype=torch.bfloat16) * 0.1
    b = torch.randn(2, 1, output_width, rank, device=device, dtype=torch.bfloat16) * 0.1
    projection.lora_a_stacked, projection.lora_b_stacked = (a,), (b,)
    backend = object.__new__(Glm5NextKPoolIndexerBackend)
    nn.Module.__init__(backend)
    backend.wq_b = projection
    inputs = torch.randn(5, width, device=device, dtype=torch.bfloat16) * 0.1
    for adapter_id in (1, 0, 2, 1):
        punica.update_metadata(LoRAMapping((adapter_id,) * 5, (adapter_id,), is_prefill=True), [1, 2], 3, 128)
        indices = punica.token_lora_indices.clone()
        actual = backend._project_queries(inputs, 2, 5)
        expected = inputs[2:5].cpu().double() @ projection.base_layer.weight.cpu().double().T
        if adapter_id:
            expected += (inputs[2:5].cpu().double() @ a[adapter_id - 1, 0].cpu().double().T) @ b[
                adapter_id - 1, 0
            ].cpu().double().T
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu().double(), expected, atol=3e-3, rtol=3e-2)
        torch.testing.assert_close(punica.token_lora_indices, indices, atol=0, rtol=0)
