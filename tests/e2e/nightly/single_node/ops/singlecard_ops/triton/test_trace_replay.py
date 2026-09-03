import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("triton")

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Trace replay kernel test requires an available Ascend NPU", allow_module_level=True)

from vllm.v1.worker.gpu.sample import trace_replay as upstream_trace_replay  # noqa: E402
import vllm_ascend.patch.worker  # noqa: E402,F401


apply_trace_tokens = upstream_trace_replay.apply_trace_tokens


DEVICE = "npu"


def _tensor(values, dtype):
    return torch.tensor(values, dtype=dtype, device=DEVICE)


def test_trace_replay_uses_request_state_step_and_idx_mapping():
    sampled = _tensor([99, 98], torch.int64)
    idx_mapping = _tensor([1, 0], torch.int32)
    trace_token_ids = _tensor(
        [
            [11, 12, 13, 14],
            [21, 22, 23, 24],
        ],
        torch.int32,
    )
    trace_len = _tensor([4, 4], torch.int32)
    total_len = _tensor([5, 3], torch.int32)
    prompt_len = _tensor([4, 1], torch.int32)

    apply_trace_tokens(
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_len,
        total_len,
        prompt_len,
    )
    torch.npu.synchronize()

    # Batch row 0 maps to state 1, step 2 -> 23; row 1 maps to state 0,
    # step 1 -> 12.
    assert torch.equal(sampled.cpu(), torch.tensor([23, 12], dtype=torch.int64))


def test_trace_replay_guards_padding_empty_and_finished_rows():
    sampled = _tensor([90, 91, 92, 93, 94], torch.int64)
    idx_mapping = _tensor([0, 1, -1, 2, 1], torch.int32)
    trace_token_ids = _tensor(
        [
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        torch.int32,
    )
    trace_len = _tensor([2, 0, 2], torch.int32)
    total_len = _tensor([2, 5, 3], torch.int32)
    prompt_len = _tensor([0, 5, 1], torch.int32)

    apply_trace_tokens(
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_len,
        total_len,
        prompt_len,
    )
    torch.npu.synchronize()

    # State 0 is past the trace, state 1 has no trace, row 2 is padding, and
    # state 2 is also past the trace. Every row must remain untouched.
    assert torch.equal(sampled.cpu(), torch.tensor([90, 91, 92, 93, 94], dtype=torch.int64))


def test_trace_replay_empty_batch_is_a_noop():
    sampled = torch.empty((0,), dtype=torch.int64, device=DEVICE)
    idx_mapping = torch.empty((0,), dtype=torch.int32, device=DEVICE)
    trace_token_ids = torch.empty((1, 1), dtype=torch.int32, device=DEVICE)
    trace_len = torch.empty((1,), dtype=torch.int32, device=DEVICE)
    total_len = torch.empty((1,), dtype=torch.int32, device=DEVICE)
    prompt_len = torch.empty((1,), dtype=torch.int32, device=DEVICE)

    apply_trace_tokens(
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_len,
        total_len,
        prompt_len,
    )

    assert sampled.numel() == 0
