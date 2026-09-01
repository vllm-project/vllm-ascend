# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()

DEVICE = "npu"
DTYPE = torch.bfloat16
HIDDEN_SIZE = 1024
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NORM_EPS = 1e-6
RATIO_CONFIG = {4: (2, 8), 128: (1, 32)}


def _build_case(
    ratio: int,
    lengths: list[int],
    start_positions: list[int],
) -> dict[str, torch.Tensor | int]:
    torch.manual_seed(ratio + sum(lengths) + sum(start_positions))
    batch_size = len(lengths)
    token_count = sum(lengths)
    coff, state_block_size = RATIO_CONFIG[ratio]
    projection_size = coff * HEAD_DIM
    state_size = 2 * projection_size
    max_position = max(start + length for start, length in zip(start_positions, lengths))
    blocks_per_request = max(1, (max_position + state_block_size - 1) // state_block_size)

    hidden_states = torch.randn(token_count, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    weight_scale = HIDDEN_SIZE**-0.5
    wkv = torch.randn(projection_size, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE) * weight_scale
    wgate = torch.randn(projection_size, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE) * weight_scale
    packed_weight = torch.cat((wkv, wgate), dim=0).contiguous()
    state_cache = torch.zeros(
        batch_size * blocks_per_request + 1,
        state_block_size,
        state_size,
        dtype=torch.float32,
        device=DEVICE,
    )
    state_block_table = (
        torch.arange(batch_size, dtype=torch.int32, device=DEVICE).unsqueeze(1) * blocks_per_request
        + torch.arange(1, blocks_per_request + 1, dtype=torch.int32, device=DEVICE)
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    start_pos = torch.tensor(start_positions, dtype=torch.int32, device=DEVICE)
    rope_rows = min(token_count, token_count // ratio + batch_size)

    return {
        "hidden_states": hidden_states,
        "wkv": wkv,
        "wgate": wgate,
        "packed_weight": packed_weight,
        "state_cache": state_cache,
        "ape": torch.randn(ratio, projection_size, dtype=torch.float32, device=DEVICE),
        "norm_weight": torch.randn(HEAD_DIM, dtype=DTYPE, device=DEVICE),
        "rope_sin": torch.randn(rope_rows, ROPE_HEAD_DIM, dtype=torch.float32, device=DEVICE),
        "rope_cos": torch.randn(rope_rows, ROPE_HEAD_DIM, dtype=torch.float32, device=DEVICE),
        "state_block_table": state_block_table,
        "cu_seqlens": cu_seqlens,
        "start_pos": start_pos,
        "coff": coff,
        "ratio": ratio,
        "valid_rows": sum(
            (start + length) // ratio - start // ratio
            for start, length in zip(start_positions, lengths)
        ),
        "rope_rows": rope_rows,
    }


def _op_kwargs(case: dict[str, torch.Tensor | int]) -> dict[str, object]:
    return {
        "state_block_table": case["state_block_table"],
        "cu_seqlens": case["cu_seqlens"],
        "seqused": None,
        "start_pos": case["start_pos"],
        "rope_head_dim": ROPE_HEAD_DIM,
        "cmp_ratio": case["ratio"],
        "coff": case["coff"],
        "norm_eps": NORM_EPS,
        "rotary_mode": 2,
        "cache_mode": 1,
    }


def _assert_max_relative_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    max_diff = (actual.float() - expected.float()).abs().max()
    scale = expected.float().abs().max().clamp_min(1e-6)
    assert (max_diff / scale).item() < 1e-2


@pytest.mark.parametrize(
    ("ratio", "lengths", "start_positions"),
    [
        pytest.param(4, [16], [0], id="c4-prefill"),
        pytest.param(128, [256], [0], id="c128-prefill"),
        pytest.param(4, [4, 4], [0, 0], id="c4-mtp"),
        pytest.param(128, [4, 4], [124, 124], id="c128-mtp"),
        pytest.param(128, [2, 2], [0, 0], id="c128-no-boundary"),
    ],
)
def test_compress_norm_rope_matches_compressor(
    ratio: int,
    lengths: list[int],
    start_positions: list[int],
):
    case = _build_case(ratio, lengths, start_positions)
    reference_state = case["state_cache"].clone()
    actual_state = case["state_cache"].clone()
    kwargs = _op_kwargs(case)

    reference = torch.ops._C_ascend.compressor(
        case["hidden_states"],
        case["wkv"],
        case["wgate"],
        reference_state,
        case["ape"],
        case["norm_weight"],
        case["rope_sin"],
        case["rope_cos"],
        **kwargs,
    )
    projected = torch.nn.functional.linear(case["hidden_states"], case["packed_weight"])
    mm_kv, mm_score = projected.chunk(2, dim=-1)
    assert mm_kv.stride(0) == projected.shape[-1]
    assert mm_score.stride(0) == projected.shape[-1]
    assert not mm_kv.is_contiguous()
    assert not mm_score.is_contiguous()
    actual = torch.ops._C_ascend.compress_norm_rope(
        mm_kv,
        mm_score,
        actual_state,
        case["ape"],
        case["norm_weight"],
        case["rope_sin"],
        case["rope_cos"],
        **kwargs,
    )
    torch.npu.synchronize()

    assert actual.shape == (case["rope_rows"], HEAD_DIM)
    valid_rows = case["valid_rows"]
    if valid_rows:
        _assert_max_relative_close(actual[:valid_rows], reference[:valid_rows])
    _assert_max_relative_close(actual_state, reference_state)


def test_compress_norm_rope_rejects_noncontiguous_last_dimension():
    case = _build_case(4, [4], [0])
    projected = torch.nn.functional.linear(case["hidden_states"], case["packed_weight"])
    mm_kv, mm_score = projected.chunk(2, dim=-1)

    with pytest.raises(RuntimeError, match="last dimension must be contiguous"):
        torch.ops._C_ascend.compress_norm_rope(
            mm_kv[:, ::2],
            mm_score[:, ::2],
            case["state_cache"],
            case["ape"][:, ::2].contiguous(),
            case["norm_weight"][: HEAD_DIM // 2],
            case["rope_sin"],
            case["rope_cos"],
            **_op_kwargs(case),
        )
