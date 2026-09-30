import pytest

from vllm_ascend.ops.triton.sfa_split_kv import choose_sfa_split_count


@pytest.mark.parametrize(
    ("query_count", "topk_count", "expected"),
    [
        (0, 2048, 1),
        (1, 128, 1),
        (1, 2048, 16),
        (2, 2048, 16),
        (4, 2048, 8),
        (8, 2048, 4),
        (16, 2048, 2),
        (32, 2048, 1),
        (1, 2000, 8),
    ],
)
def test_choose_sfa_split_count(query_count, topk_count, expected):
    assert choose_sfa_split_count(query_count, topk_count) == expected
