import math

import torch

import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm_ascend.ops.gbsa_qsa import qsa_gbsa_attention


ROWS = 2
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 128


def make_expanded_indices() -> torch.Tensor:
    groups = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="npu")
    offsets = torch.arange(4, dtype=torch.int32, device="npu")
    expanded = (groups.unsqueeze(-1) * 4 + offsets).flatten(1)
    residual_slots = torch.full((ROWS, 3), -1, dtype=torch.int32, device="npu")
    return torch.cat((expanded, residual_slots), dim=1)


def reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for row in range(query.shape[0]):
        request = int(token_to_req[row])
        length = int(seq_lens[request])
        tokens = torch.arange(length, device=query.device)
        pages = block_table[request, tokens // PAGE_SIZE].long()
        offsets = tokens % PAGE_SIZE
        keys = key_cache[pages, offsets].float().repeat_interleave(2, dim=1)
        values = value_cache[pages, offsets].float().repeat_interleave(2, dim=1)
        scores = torch.einsum("hd,shd->hs", query[row].float(), keys)
        probabilities = (scores / math.sqrt(HEAD_DIM)).softmax(-1)
        rows.append(torch.einsum("hs,shd->hd", probabilities, values))
    return torch.stack(rows)


def main() -> None:
    torch.manual_seed(20261009)
    query = torch.randn((ROWS, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    key_cache = torch.randn((4, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    value_cache = torch.randn_like(key_cache)
    logical_indices = make_expanded_indices()
    block_table = torch.tensor([[2], [3]], dtype=torch.int32, device="npu")
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([9, 11], dtype=torch.int32, device="npu")
    output = torch.empty_like(query)

    def call() -> torch.Tensor:
        return qsa_gbsa_attention(
            query,
            key_cache,
            value_cache,
            logical_indices,
            block_table,
            token_to_req,
            seq_lens,
            output,
        )

    expected = reference(query, key_cache, value_cache, block_table, token_to_req, seq_lens)
    eager = call().clone()
    torch.npu.synchronize()
    torch.testing.assert_close(eager.float(), expected, rtol=2e-2, atol=2e-2)
    print("PASS eager")

    compiled_call = torch.compile(call, backend="npugraph_ex", fullgraph=True, dynamic=False)
    compiled = compiled_call().clone()
    torch.npu.synchronize()
    torch.testing.assert_close(compiled.float(), expected, rtol=2e-2, atol=2e-2)
    print("PASS compile")

    call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured_output = call()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured_output.float(), expected, rtol=2e-2, atol=2e-2)
    print("PASS capture+replay-1")

    query.copy_(query * 0.5)
    expected_replay = reference(query, key_cache, value_cache, block_table, token_to_req, seq_lens)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured_output.float(), expected_replay, rtol=2e-2, atol=2e-2)
    print("PASS replay-2-updated-input")


if __name__ == "__main__":
    main()
