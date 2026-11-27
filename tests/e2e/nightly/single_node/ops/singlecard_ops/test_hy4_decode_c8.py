# SPDX-License-Identifier: Apache-2.0
"""C8 Decode and paged IndexPool checks on Ascend 950 with CANN 9.3."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("cann_ops_transformer")

from vllm_ascend.device.device_config import is_950  # noqa: E402
from vllm_ascend.hisparse.c8 import dequantize_c8_rows  # noqa: E402
from vllm_ascend.models.hy_v4_flash.c8_decode import attend_compact_c8  # noqa: E402
from vllm_ascend.models.hy_v4_flash.index_pool_cache import copy_pool_keys  # noqa: E402
from vllm_ascend.models.hy_v4_flash.index_pool_decode import refresh_pages, select_pages, write_pages  # noqa: E402
from vllm_ascend.models.hy_v4_flash.index_pool_select import select_pool_candidates, write_pool_scores  # noqa: E402


@pytest.fixture(autouse=True)
def npu_device():
    if not torch.npu.is_available() or not is_950():
        pytest.skip("Requires Ascend 950 and the HY4 CANN operators")
    torch.npu.set_device(0)
    torch.manual_seed(1024)


@pytest.mark.parametrize(
    "rows,heads,width", [(6, 32, 2048), (24, 32, 2048), (6, 64, 512), (18, 64, 512), (24, 64, 512), (48, 64, 512)]
)
def test_c8_graph_replay_matches_fp32_attention(rows, heads, width):
    query = torch.randn(rows, heads, 576, device="npu", dtype=torch.bfloat16) * 0.1
    latent = torch.randn(rows, width, 512, device="npu", dtype=torch.float32) * 0.1
    rope = torch.randn(rows, width, 64, device="npu", dtype=torch.bfloat16) * 0.1
    scales = torch.pow(
        2.0, torch.ceil(torch.log2(latent.view(rows, width, 4, 128).abs().amax(-1).clamp_min(1e-4) / 448))
    )
    payload = (latent / scales.repeat_interleave(128, -1)).to(torch.float8_e4m3fn)
    packed = torch.cat((payload.view(torch.uint8), rope.view(torch.uint8), scales.contiguous().view(torch.uint8)), -1)
    sink = torch.randn(heads, device="npu", dtype=torch.float32)
    selected = torch.arange(width, device="npu").expand(rows, -1).clone()
    scale = 576**-0.5
    attend_compact_c8(query, packed, selected, sink, scale)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = attend_compact_c8(query, packed, selected, sink, scale)

    # Change live lengths after capture, including an entirely padded row.
    for counts in (
        [width - 3, width - 2, width - 1, width, width - 3, width - 2],
        [0, 1, min(511, width), min(512, width), width, 2],
    ):
        lengths = torch.tensor(counts * (rows // 6), device="npu")
        selected.copy_(torch.arange(width, device="npu").expand(rows, -1))
        selected.masked_fill_(torch.arange(width, device="npu")[None] >= lengths[:, None], -1)
        graph.replay()
        values = dequantize_c8_rows(packed)
        scores = torch.bmm(query.float(), values.transpose(1, 2)) * scale
        scores.masked_fill_(selected[:, None] < 0, -torch.inf)
        probabilities = torch.cat((scores, sink[None, :, None].expand(rows, -1, -1)), -1).softmax(-1)[..., :-1]
        expected = torch.bmm(probabilities, values[..., :512])
        relative_l2 = (actual.float() - expected).norm() / expected.norm()
        assert torch.isfinite(actual).all()
        assert relative_l2 < 0.01
        assert torch.count_nonzero(actual[lengths == 0]) == 0


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_paged_indexer_replays_short_long_reused_and_updated_pages(batch):
    groups = 65536
    tables = torch.stack([torch.randperm(groups // 16, device="npu").int() for _ in range(batch)])
    cache = torch.zeros(groups * 4, 1000, device="npu", dtype=torch.uint8)
    keys = torch.randn(groups, 128, device="npu", dtype=torch.bfloat16)
    copy_pool_keys(cache, tables[0], keys, 0, 64, 4, write=True)
    pages = torch.zeros(groups // 16, 16, 1, 128, device="npu", dtype=torch.bfloat16)
    query = torch.randn(batch * 6, 16, 128, device="npu", dtype=torch.bfloat16).float()
    weights = torch.rand(batch * 6, 16, device="npu")
    visible = torch.stack([torch.arange(133121 + i, 133127 + i, device="npu") for i in range(batch)])
    for i in range(batch):
        refresh_pages(cache, pages, tables[i], 64, 133127 + i)
    select_pages(query, weights, pages, tables, visible, groups)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = select_pages(query, weights, pages, tables, visible, groups)

    cases = [
        [133121 + i for i in range(batch)],
        [0, 200, 507, 1000, 1500, 2000, -6, 2040][:batch],
        [2043, -6, 2100, 10, 133121, 2044, 4096, 2047][:batch],
    ]
    for starts in cases:
        # Reused physical pages must reflect the incoming request's cache.
        keys.mul_(-0.5)
        copy_pool_keys(cache, tables[0], keys, 0, 64, 4, write=True)
        for i, start in enumerate(starts):
            visible[i].copy_(torch.arange(start, start + 6, device="npu"))
            refresh_pages(cache, pages, tables[i], 64, max(start + 6, 0))
        # Mirror a newly completed group into both packed and derived layouts.
        first = torch.tensor(max(starts[0], 0) // 4, device="npu")
        end = first + 2
        updated = torch.randn(2, 128, device="npu", dtype=torch.bfloat16)
        copy_pool_keys(cache, tables[0], updated, first, 64, 4, write=True, end_group=end)
        write_pages(updated, pages, tables[0], first, end, 64)
        graph.replay()
        for i in range(batch):
            scores = write_pool_scores(
                query[i * 6 : (i + 1) * 6],
                weights[i * 6 : (i + 1) * 6],
                cache,
                tables[i],
                visible[i],
                64,
                groups,
                4,
            )
            expected = select_pool_candidates(scores, visible[i], 4, 511)
            assert torch.equal(actual[i].sort(-1).values, expected.sort(-1).values)
