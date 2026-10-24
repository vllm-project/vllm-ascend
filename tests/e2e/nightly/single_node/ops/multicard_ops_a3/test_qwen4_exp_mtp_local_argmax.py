# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen4Exp draft argmax and collective payloads on two NPUs."""

import random
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_npu
from torch import nn
from vllm.distributed import parallel_state
from vllm.model_executor.layers import logits_processor as logits_module

from vllm_ascend.models.qwen4_exp.mtp import AscendQwen4ExpMTP
from vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessor, VocabParallelMode

TP_SIZE = 2
PADDED_VOCAB_SIZE = 131_072
VOCAB_SIZE = PADDED_VOCAB_SIZE - 1


class LinearHeadMethod:
    def apply(self, layer, hidden_states, bias=None):
        return F.linear(hidden_states, layer.weight, bias)


def _make_model(rank, dtype):
    weight = torch.zeros(PADDED_VOCAB_SIZE, 4, dtype=dtype, device="npu")
    weight[3, 0] = weight[PADDED_VOCAB_SIZE // 2 + 7, 0] = 4
    weight[PADDED_VOCAB_SIZE // 2 + 9, 1] = 5
    weight[VOCAB_SIZE - 1, 2] = 6
    weight[-1] = 100  # Padding must not win even on the last TP rank.
    head = nn.Module()
    head.weight = nn.Parameter(weight.chunk(TP_SIZE)[rank].contiguous(), requires_grad=False)
    head.tp_size = TP_SIZE
    head.parallel_mode = VocabParallelMode.STANDARD
    head.quant_method = LinearHeadMethod()
    head.shard_indices = SimpleNamespace(
        org_vocab_start_index=rank * (PADDED_VOCAB_SIZE // TP_SIZE),
        num_org_vocab_padding=rank,
    )
    processor = AscendLogitsProcessor.__new__(AscendLogitsProcessor)
    nn.Module.__init__(processor)
    processor.head_dtype = None
    processor.org_vocab_size = VOCAB_SIZE
    processor.scale = 0.5
    processor.soft_cap = 4.0
    processor.logits_as_input = False
    processor.use_all_gather = True
    model = AscendQwen4ExpMTP.__new__(AscendQwen4ExpMTP)
    nn.Module.__init__(model)
    model.lm_head = head
    model.logits_processor = processor
    return model


@torch.inference_mode()
def _worker(rank, port, result_queue):
    group = None
    try:
        torch_npu.npu.set_device(rank)
        parallel_state.init_distributed_environment(
            world_size=TP_SIZE,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="hccl",
        )
        group = parallel_state.init_model_parallel_group(
            [[0, 1]],
            local_rank=rank,
            backend="hccl",
            group_name="qwen4_exp_mtp_local_argmax",
            use_device_communicator=True,
        )
        all_gather = logits_module.tensor_model_parallel_all_gather
        calls = []

        def tracked_all_gather(tensor, dim=-1):
            calls.append((tuple(tensor.shape), tensor.dtype))
            return all_gather(tensor, dim=dim)

        with (
            patch.object(parallel_state, "_TP", group),
            patch.object(logits_module, "tensor_model_parallel_all_gather", tracked_all_gather),
        ):
            for dtype in (torch.float32, torch.bfloat16):
                model = _make_model(rank, dtype)
                for batch_size in (1, 8, 32):
                    hidden = torch.eye(4, dtype=dtype, device="npu").repeat(batch_size, 1)[:batch_size].contiguous()
                    calls.clear()
                    expected = model.compute_logits(hidden).argmax(dim=-1)
                    assert calls == [((batch_size, PADDED_VOCAB_SIZE // TP_SIZE), dtype)]
                    calls.clear()
                    actual = model.get_top_tokens(hidden)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    assert calls == [((batch_size, 2), torch.float32)]

                    for _ in range(3):
                        model.get_top_tokens(hidden)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        captured = model.get_top_tokens(hidden)
                    # Replay changes the winning token without changing any
                    # device address, exercising the real HCCL graph path.
                    for _ in range(2):
                        hidden.copy_(hidden.roll(1, dims=-1))
                        graph.replay()
                        expected = model.compute_logits(hidden).argmax(dim=-1)
                        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        result_queue.put(None)
    except Exception:
        result_queue.put(traceback.format_exc())
    finally:
        if group is not None:
            group.destroy()
        parallel_state.destroy_distributed_environment()


def test_qwen4_exp_mtp_local_argmax_tp2_eager_and_graph():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = 29_501 + random.randint(0, 10_000)
    processes = [context.Process(target=_worker, args=(rank, port, result_queue)) for rank in range(TP_SIZE)]
    try:
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=300) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        assert all(process.exitcode == 0 for process in processes)
        assert results == [None] * TP_SIZE, "\n".join(result for result in results if result is not None)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
