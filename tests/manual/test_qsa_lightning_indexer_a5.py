import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.models.qwen4_exp.lightning_indexer import (
    qsa_select_paged_tokens_lightning,
)
from vllm_ascend.models.qwen4_exp.ops import qsa_select_paged_tokens


def main() -> None:
    torch.manual_seed(20261009)
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    num_pages = 6
    page_size = 384
    num_heads = 4
    head_dim = 128
    sequence_length = 4096
    token_topk = 2048
    compress_ratio = 4

    query = torch.randn((1, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        (num_pages, page_size, 1, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=device)
    query_positions = torch.tensor([sequence_length - 1], dtype=torch.int64, device=device)
    sequence_lengths = torch.tensor([sequence_length], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)

    expected = qsa_select_paged_tokens(
        query,
        key_cache,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )
    actual = qsa_select_paged_tokens_lightning(
        query,
        key_cache,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        query_start_loc,
        token_topk,
        compress_ratio,
        None,
        use_e3=False,
    )
    torch.npu.synchronize()
    block_topk = token_topk // compress_ratio
    expected_groups = expected[:, :block_topk].sort(dim=-1).values
    actual_groups = actual[:, :block_topk].sort(dim=-1).values
    torch.testing.assert_close(actual_groups.cpu(), expected_groups.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual[:, block_topk:].cpu(), expected[:, block_topk:].cpu(), rtol=0, atol=0)
    print("PASS A5 LI-v2 selector matches the Torch reference")


if __name__ == "__main__":
    main()
