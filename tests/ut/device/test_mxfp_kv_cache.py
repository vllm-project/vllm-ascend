import torch

from vllm_ascend.device.mxfp_kv_cache import scatter_mxfp_k_scale_cache


def test_scatter_mxfp_k_scale_cache_ignores_full_graph_padding():
    key_scale_cache = torch.full((1, 1, 2, 1, 2), 7, dtype=torch.uint8)
    key_scale = torch.tensor(
        [
            [[[[3, 4]]]],
            [[[[9, 9]]]],
        ],
        dtype=torch.uint8,
    ).view(2, 1, 1, 2)
    slot_mapping = torch.tensor([1, -1], dtype=torch.int64)

    scatter_mxfp_k_scale_cache(
        key_scale,
        key_scale_cache,
        slot_mapping,
        block_size=2,
    )

    torch.testing.assert_close(
        key_scale_cache[0, 0, 0],
        torch.tensor([[7, 7]], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        key_scale_cache[0, 0, 1],
        torch.tensor([[3, 4]], dtype=torch.uint8),
    )
