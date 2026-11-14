import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.attention_update import prepare_attention_update
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

_REQUIRES_A2_A3 = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Requires Ascend A2/A3",
)

DEVICE = "npu"


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    init_device_properties_triton()


def _ref_prepare_attention_update(
    lses: list[torch.Tensor],
    previous_anchor: torch.Tensor | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """PyTorch baseline for prepare_attention_update (LSE normalization).

    anchor = max(previous_anchor_or_lses[0], lses[1..], lses[2..], ...),
    where ±inf LSE entries are excluded from the max (treated as -inf).
    When previous_anchor is given, lses[0] does NOT participate in the anchor
    and its relative output is ``lses[0] + (previous_anchor - anchor)``;
    otherwise every group's relative output is ``lses[i] - anchor``.
    """
    rows = lses[0].numel()
    if rows == 0:
        return tuple(t.new_empty(0) for t in lses), lses[0].new_empty(0)

    stacked = torch.stack(lses, dim=0)  # [COUNT, rows], fp32 on device
    neg_inf = torch.tensor(-float("inf"), dtype=stacked.dtype, device=stacked.device)

    if previous_anchor is not None:
        prev = previous_anchor
        rest = stacked[1:]  # lses[1..]
        rest_eff = torch.where(rest.abs() < float("inf"), rest, neg_inf)
        anchor = torch.maximum(prev, rest_eff.max(dim=0).values)
        relative0 = stacked[0] + (prev - anchor)
    else:
        eff = torch.where(stacked.abs() < float("inf"), stacked, neg_inf)
        anchor = eff.max(dim=0).values
        relative0 = stacked[0] - anchor

    relative = stacked - anchor  # [COUNT, rows]
    relative = relative.clone()
    relative[0] = relative0
    return tuple(t for t in relative), anchor


# ---------------------------------------------------------------------------
# Correctness: no previous_anchor (HAS_ANCHOR=False).
#   anchor = max(lses), ±inf excluded; relative[i] = lses[i] - anchor.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@pytest.mark.parametrize(
    ("rows", "count"),
    [
        pytest.param(128, 2, id="rows-128-count-2"),
        pytest.param(257, 3, id="rows-257-tail-count-3"),
        pytest.param(1, 2, id="rows-1"),
        pytest.param(1024, 4, id="rows-1024-count-4"),
    ],
)
@torch.inference_mode()
def test_prepare_attention_update_no_anchor(rows, count):
    torch.manual_seed(0)
    lses = [torch.randn(rows, dtype=torch.float32, device=DEVICE) for _ in range(count)]
    out_rel, out_anchor = prepare_attention_update(lses, None)
    ref_rel, ref_anchor = _ref_prepare_attention_update(lses, None)

    assert len(out_rel) == count
    assert out_anchor.shape == (rows,)
    torch.testing.assert_close(out_anchor.cpu(), ref_anchor.cpu(), rtol=1e-5, atol=1e-5)
    for i in range(count):
        assert out_rel[i].shape == (rows,)
        torch.testing.assert_close(out_rel[i].cpu(), ref_rel[i].cpu(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Correctness: with previous_anchor (HAS_ANCHOR=True).
#   anchor = max(previous_anchor, lses[1..]); lses[0] excluded from anchor;
#   relative[0] = lses[0] + (previous - anchor); relative[i] = lses[i] - anchor.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@pytest.mark.parametrize(
    ("rows", "count"),
    [
        pytest.param(128, 2, id="rows-128-count-2"),
        pytest.param(257, 3, id="rows-257-tail-count-3"),
        pytest.param(1, 2, id="rows-1"),
        pytest.param(512, 5, id="rows-512-count-5"),
    ],
)
@torch.inference_mode()
def test_prepare_attention_update_with_anchor(rows, count):
    torch.manual_seed(1)
    lses = [torch.randn(rows, dtype=torch.float32, device=DEVICE) for _ in range(count)]
    previous_anchor = torch.randn(rows, dtype=torch.float32, device=DEVICE)

    out_rel, out_anchor = prepare_attention_update(lses, previous_anchor)
    ref_rel, ref_anchor = _ref_prepare_attention_update(lses, previous_anchor)

    assert len(out_rel) == count
    assert out_anchor.shape == (rows,)
    torch.testing.assert_close(out_anchor.cpu(), ref_anchor.cpu(), rtol=1e-5, atol=1e-5)
    for i in range(count):
        assert out_rel[i].shape == (rows,)
        torch.testing.assert_close(out_rel[i].cpu(), ref_rel[i].cpu(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Correctness: ±inf LSE entries are excluded from the anchor.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@torch.inference_mode()
def test_prepare_attention_update_inf_excluded_from_anchor():
    rows = 64
    lses = [
        torch.randn(rows, dtype=torch.float32, device=DEVICE),
        torch.randn(rows, dtype=torch.float32, device=DEVICE),
    ]
    # Make row 0 of lses[1] +inf; it must not become the anchor there.
    lses[1][0] = float("inf")

    out_rel, out_anchor = prepare_attention_update(lses, None)
    ref_rel, ref_anchor = _ref_prepare_attention_update(lses, None)

    torch.testing.assert_close(out_anchor.cpu(), ref_anchor.cpu(), rtol=1e-5, atol=1e-5)
    for i in range(2):
        torch.testing.assert_close(out_rel[i].cpu(), ref_rel[i].cpu(), rtol=1e-5, atol=1e-5)
    # Sanity: the anchor at row 0 should be lses[0][0], not +inf.
    assert not torch.isinf(out_anchor[0]).any()


# ---------------------------------------------------------------------------
# Boundary: rows == 0 short-circuits (returns empty tensors, no kernel launch).
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@torch.inference_mode()
def test_prepare_attention_update_empty_rows():
    lses = [torch.empty(0, dtype=torch.float32, device=DEVICE) for _ in range(2)]
    out_rel, out_anchor = prepare_attention_update(lses, None)

    assert len(out_rel) == 2
    assert out_anchor.shape == (0,)
    for r in out_rel:
        assert r.shape == (0,)


# ---------------------------------------------------------------------------
# Robustness: the entry guards input contracts so mismatched inputs surface
# as a clear error instead of crashing the kernel or silently computing on the
# wrong elements.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@torch.inference_mode()
def test_prepare_attention_update_rejects_empty_lses():
    with pytest.raises(AssertionError, match="non-empty"):
        prepare_attention_update([], None)


@_REQUIRES_A2_A3
@torch.inference_mode()
def test_prepare_attention_update_rejects_too_many_lses():
    lses = [torch.randn(8, dtype=torch.float32, device=DEVICE) for _ in range(17)]
    with pytest.raises(AssertionError, match="16 LSE inputs"):
        prepare_attention_update(lses, None)


@_REQUIRES_A2_A3
@torch.inference_mode()
def test_prepare_attention_update_rejects_mismatched_numel():
    lses = [
        torch.randn(16, dtype=torch.float32, device=DEVICE),
        torch.randn(8, dtype=torch.float32, device=DEVICE),
    ]
    with pytest.raises(AssertionError, match="lses\\[1\\] has 8 elements"):
        prepare_attention_update(lses, None)


@_REQUIRES_A2_A3
@torch.inference_mode()
def test_prepare_attention_update_rejects_mismatched_dtype():
    lses = [
        torch.randn(16, dtype=torch.float32, device=DEVICE),
        torch.randn(16, dtype=torch.float16, device=DEVICE),
    ]
    with pytest.raises(AssertionError, match="lses\\[1\\] dtype"):
        prepare_attention_update(lses, None)


@_REQUIRES_A2_A3
@torch.inference_mode()
def test_prepare_attention_update_rejects_mismatched_device():
    lses = [
        torch.randn(16, dtype=torch.float32, device=DEVICE),
        torch.randn(16, dtype=torch.float32, device="cpu"),
    ]
    with pytest.raises(AssertionError, match="lses\\[1\\] device"):
        prepare_attention_update(lses, None)
