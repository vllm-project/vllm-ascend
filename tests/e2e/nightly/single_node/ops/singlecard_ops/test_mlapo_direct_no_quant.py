import pytest
import torch
import torch_npu

from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.utils import enable_custom_op

from . import test_mlapo_direct_modes as qdown

mapped_offload = qdown.mapped_offload

enable_custom_op()


@pytest.mark.parametrize("enable_rope", [True, False])
@pytest.mark.parametrize("token_num", [1, 6, 128])
@torch.inference_mode()
def test_mla_preprocess_kernel(enable_rope, token_num, monkeypatch):
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    _exercise_no_quant_case(enable_rope, token_num)


@pytest.mark.parametrize("enable_rope", [True, False])
@pytest.mark.parametrize("writeback_enabled", [True, False], ids=["writeback", "peer"])
@torch.inference_mode()
def test_mla_no_quant_mapped_writeback_graph(enable_rope, writeback_enabled, mapped_offload, monkeypatch):
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    _exercise_no_quant_case(enable_rope, 6, mapped_offload, writeback_enabled)


def _exercise_no_quant_case(enable_rope, token_num, offload=None, writeback_enabled=True):
    torch.manual_seed(0)
    # Retain the original no_quant test's dimensions. N=6 exercises several
    # destinations; N=128 also exercises per-core loops and local-buffer reuse.
    head_num, hidden_size, q_lora_rank = 2, 7168, 1536
    kv_lora_rank, rope_dim, qk_nope_head_dim = 512, 64, 128
    block_size, dtype = 128, torch.bfloat16
    mm1_out = q_lora_rank + kv_lora_rank + rope_dim
    mm2_out = head_num * (qk_nope_head_dim + rope_dim)
    hidden = torch.randn(token_num, hidden_size, dtype=dtype, device="npu") * 0.1
    wdqkv = torch.randn(1, hidden_size // 16, mm1_out, 16, dtype=dtype, device="npu") * 0.01
    wuq = torch.randn(1, q_lora_rank // 16, mm2_out, 16, dtype=dtype, device="npu") * 0.01
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    cos = torch.rand(token_num, 1, 1, rope_dim, dtype=dtype, device="npu")
    sin = torch.rand_like(cos)
    inputs = (
        hidden,
        wdqkv,
        None,
        torch.ones(q_lora_rank, dtype=dtype, device="npu"),
        None,
        wuq,
        None,
        torch.ones(kv_lora_rank, dtype=dtype, device="npu"),
        cos.view(token_num, rope_dim) if enable_rope else None,
        sin.view(token_num, rope_dim) if enable_rope else None,
        # Original no_quant dispatch uses ND W_UK and ND krope_ctkv caches.
        torch.randn(head_num, qk_nope_head_dim, kv_lora_rank, dtype=dtype, device="npu") * 0.01,
    )
    # The schema accepts None for these tensors but has no default values.
    quant_args = {
        name: None
        for name in (
            "quant_scale0",
            "quant_offset0",
            "bias0",
            "quant_scale1",
            "quant_offset1",
            "bias1",
            "ctkv_scale",
            "q_nope_scale",
        )
    }
    caches = tuple(
        torch.full((1, block_size, width), float("nan"), dtype=dtype, device="npu")
        for width in (kv_lora_rank, rope_dim)
    )
    compact_slots = torch.arange(token_num, dtype=torch.int32, device="npu")
    q_outputs = tuple(
        torch.full(shape, float("nan"), dtype=dtype, device="npu")
        for shape in ((token_num, head_num, kv_lora_rank), (token_num, head_num, rope_dim), (token_num, q_lora_rank))
    )
    reference = (*q_outputs, caches[0].view(-1, kv_lora_rank)[:token_num], caches[1].view(-1, rope_dim)[:token_num])

    def original(inner=True):
        torch.ops._C_ascend.mla_preprocess(
            *inputs,
            *caches,
            compact_slots,
            **quant_args,
            cache_mode="krope_ctkv",
            quant_mode="no_quant",
            enable_inner_out=inner,
            q_out0=q_outputs[0],
            kv_cache_out0=caches[0],
            q_out1=q_outputs[1],
            kv_cache_out1=caches[1],
            inner_out=q_outputs[2],
        )

    original(inner=False)
    torch.npu.synchronize()
    original_outputs = [reference[index].clone() for index in (0, 1, 3, 4)]
    for value in reference:
        value.fill_(float("nan"))
    original()
    torch.npu.synchronize()
    # Exposing q_c must not change the four outputs of the original branch.
    for index, expected in zip((0, 1, 3, 4), original_outputs):
        qdown._assert_same_bits(reference[index], expected, f"no_quant original inner on/off output {index}")

    outputs = tuple(torch.full_like(value, float("nan")) for value in reference)

    direct_inputs = (*inputs[:8], cos if enable_rope else None, sin if enable_rope else None, *inputs[10:])

    def direct(writeback=None, call_inputs=direct_inputs):
        torch.ops._C_ascend.mla_preprocess_with_kv_offload(
            *call_inputs,
            **quant_args,
            quant_mode="no_quant",
            q_out0=outputs[0],
            q_out1=outputs[1],
            inner_out=outputs[2],
            current_k=outputs[3],
            current_rope=outputs[4],
            **(writeback or {}),
        )

    def check_outputs():
        torch.npu.synchronize()
        for name, actual, expected in zip(("q_nope", "q_rope", "q_c", "current_k", "current_rope"), outputs, reference):
            qdown._assert_same_bits(actual, expected, name)

    # Preserve omitted-slots/2D-RoPE calls while checking metadata slots and
    # original 4D cos/sin with the zero-default CPU writeback destinations.
    direct(call_inputs=inputs)
    check_outputs()
    direct({"writeback_slots": compact_slots})
    check_outputs()
    if offload is None:
        return

    # Guard rows surround the advertised pool capacity so negative and
    # capacity-boundary slots cannot silently corrupt adjacent storage.
    pools = tuple(
        offload.empty([block_size + 2, width], dtype=dtype, pin_memory=True) for width in (kv_lora_rank, rope_dim)
    )
    slots = torch.empty(token_num, dtype=torch.int32, device="npu")
    writeback = {
        "writeback_slots": slots,
        "writeback_k_base": pools[0][1:-1].data_ptr() if writeback_enabled else 0,
        "writeback_rope_base": pools[1][1:-1].data_ptr() if writeback_enabled else 0,
        "writeback_capacity": block_size if writeback_enabled else 0,
    }
    # Use the real CPU-to-NPU offload consumer on a different stream, after
    # the producer's completion event, including both guard rows in the read.
    readback = tuple(torch.empty(value.shape, dtype=dtype, device="npu") for value in pools)
    src = torch.tensor([value.data_ptr() for value in pools], dtype=torch.int64, device="npu")
    dst = torch.tensor([value.data_ptr() for value in readback], dtype=torch.int64, device="npu")
    sizes = torch.tensor([value.numel() * value.element_size() for value in pools], dtype=torch.int32, device="npu")
    count = torch.tensor(2, dtype=torch.int32, device="npu")
    consumer, done = torch.npu.Stream(), torch.npu.Event()
    slot_ids = []

    def update(iteration):
        torch.manual_seed(100 + iteration)
        hidden.copy_(torch.randn_like(hidden) * 0.1)
        cos.copy_(torch.rand_like(cos))
        sin.copy_(torch.rand_like(sin))
        slot_ids[:] = [3 + iteration, 7 + iteration, -1, 11 + iteration, 15 + iteration, block_size]
        slots.copy_(torch.tensor(slot_ids, dtype=slots.dtype, device=slots.device))
        for pool in pools:
            pool.fill_(-17)
        for value in (*outputs, *reference):
            value.fill_(float("nan"))
        original()
        torch.npu.synchronize()

    def launch():
        direct(writeback)
        producer = torch.npu.current_stream()
        done.record(producer)
        consumer.wait_event(done)
        with torch.npu.stream(consumer):
            assert offload.sparse_copy(src, dst, sizes, count, hidden.device) in (None, 0)
        producer.wait_stream(consumer)

    def check():
        check_outputs()
        for pool, read, value in zip(pools, readback, reference[3:]):
            expected = torch.full_like(pool, -17)
            value = value.cpu()
            for row, slot in enumerate(slot_ids):
                if writeback_enabled and 0 <= slot < block_size:
                    expected[slot + 1] = value[row]
            qdown._assert_same_bits(pool, expected, "CPU pool including guard and untouched rows")
            qdown._assert_same_bits(read, expected, "CPU-to-NPU consumer")

    update(0)
    launch()
    check()
    streams = (torch.npu.Stream(), torch.npu.Stream())
    graphs = []
    for stream in streams:
        graph = UpdatableGraph()
        with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
            launch()
        graphs.append(graph)
        check()
    prior_q = outputs[0].cpu().clone()
    for iteration in (1, 2, 3):
        update(iteration)
        with torch.npu.stream(streams[iteration % 2]):
            graphs[(iteration + 1) % 2].replay()
        # Fresh eager results prevent two stale captured outputs from passing.
        check()
        assert not torch.equal(outputs[0].cpu().view(torch.int16), prior_q.view(torch.int16))
        prior_q = outputs[0].cpu().clone()
