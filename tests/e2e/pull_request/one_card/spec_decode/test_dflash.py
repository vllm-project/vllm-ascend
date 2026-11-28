from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationConfig

from tests.e2e.pull_request.one_card.spec_decode.utils import DFLASH
from tests.e2e.pull_request.utils import SPEC_DECODE_PROMPTS, _run_speculative_decoding
from vllm_ascend.attention.a3_mla_sink import A3MLAAdapter, A3MLAConfig, get_a3_mla_ops, supports_a3_mla_window
from vllm_ascend.attention.flashmla import FlashMLAMetadataBuilder
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

EXPECTED_ACCEPTANCE_LENGTH_STATIC = 2.80


@pytest.mark.parametrize("method", DFLASH.keys())
@pytest.mark.parametrize("num_speculative_tokens", [8])
def test_dflash_acceptance(
    method: str,
    num_speculative_tokens: int,
):
    main_model_name = DFLASH[method]["main"]
    spec_model_name = DFLASH[method]["spec"]
    speculative_config = {
        "method": "dflash",
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
    }
    num_prompts = len(SPEC_DECODE_PROMPTS)
    capture_sizes = [num_prompts * (num_speculative_tokens + 1)]
    cudagraph_mode = "FULL_DECODE_ONLY"

    _run_speculative_decoding(
        model_name=main_model_name,
        speculative_config=speculative_config,
        expected_acceptance_length=EXPECTED_ACCEPTANCE_LENGTH_STATIC,
        runner_kwargs={
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "distributed_executor_backend": "mp",
            "gpu_memory_utilization": 0.8,
            "compilation_config": CompilationConfig(
                cudagraph_mode=cudagraph_mode,
                cudagraph_capture_sizes=capture_sizes,
            ),
            "enable_prefix_caching": False,
        },
        example_prompts=SPEC_DECODE_PROMPTS,
        max_tokens=256,
        is_moe=False,
        enable_thinking=False,
    )


@torch.inference_mode()
def test_a3_dflash_mla_window_replay():
    if get_ascend_device_type() != AscendDeviceType.A3 or not supports_a3_mla_window() or get_a3_mla_ops() is None:
        pytest.skip("requires the A3 windowed MLA provider")
    device = torch.device("npu")
    heads, window, limit, width = 12, 1024, 1031, 8
    adapter = A3MLAAdapter.load(A3MLAConfig(heads, 1.0, mask_mode=4, sliding_window=window))
    impl = SimpleNamespace(
        a3_mla_adapter=adapter,
        dtype=torch.bfloat16,
        use_mla_rope=False,
        use_quant_flashmla=False,
        vllm_config=SimpleNamespace(speculative_config=object()),
    )
    builder = FlashMLAMetadataBuilder(impl, device, 3, None)
    pages = (limit + 127) // 128
    cache, rope = adapter.allocate_cache(2 * pages, device=device)
    cache.zero_()
    rope.zero_()
    # Zero queries give every visible key equal weight. Mark the left edge and
    # distinguish each request's KV pages so a stale table or window is visible.
    markers = torch.tensor([0, 1, 3, 4, 7, 1023, 1027, 1030])
    for req in range(2):
        for lane, key in enumerate(markers.tolist()):
            cache[req * pages + key // 128, key % 128, 0, lane] = window * (req + 1)
    query = torch.zeros(3 * width, heads, 576, dtype=torch.bfloat16, device=device)
    common = SimpleNamespace(
        num_actual_tokens=2 * width,
        num_input_tokens=3 * width,
        causal=True,
        seq_lens=torch.empty(2, device=device, dtype=torch.int32),
        query_start_loc=torch.arange(3, device=device, dtype=torch.int32) * width,
        block_table_tensor=torch.arange(2 * pages, device=device, dtype=torch.int32).reshape(2, pages),
        slot_mapping=torch.arange(2 * width, device=device),
        positions=torch.arange(2 * width, device=device),
    )

    def refresh(starts, capture=False):
        first = torch.tensor(starts, device=device, dtype=torch.int32)
        common.seq_lens.copy_((first + width).clamp_max(limit))
        with builder.limit_query_positions(first, limit):
            return builder.build(common, 2, 2 * width, False, retain_for_graph=capture)

    batch = refresh([1023, 1019], capture=True)

    def attention():
        output, _ = batch.adapter.attention(
            query,
            (cache, rope),
            block_table=batch.block_table,
            cache_seqlens=batch.cache_lens,
            cu_seqlens_q=batch.cu,
            metadata=batch.schedule,
            output_mask=batch.output_mask,
            return_softmax_lse=False,
        )
        return output

    # The runner warms the provider before it records a graph.
    attention()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = attention()
    for starts in ([1027, 1020], [1017, 1029], [1023, 1031]):
        assert refresh(starts) is batch
        graph.replay()
        expected = torch.zeros(3 * width, len(markers))
        live = torch.zeros(3 * width, dtype=torch.bool)
        for req, first in enumerate(starts):
            for offset in range(min(width, limit - first)):
                pos = first + offset
                begin = max(0, pos - window + 1)
                row = req * width + offset
                live[row] = True
                expected[row] = (markers >= begin) * (markers <= pos) * (window * (req + 1) / (pos - begin + 1))
        torch.testing.assert_close(
            output[:, :, : len(markers)].float().cpu(), expected.expand(heads, -1, -1), atol=0.01, rtol=0
        )
        assert torch.equal(batch.slots.cpu() >= 0, live)
