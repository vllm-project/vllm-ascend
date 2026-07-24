import torch

from vllm_ascend.spec_decode.dflash_debug import _next_trace_call, tensor_summary


def test_next_trace_call_is_initial_then_power_of_two():
    owner = object_with_attributes = type("TraceOwner", (), {})()

    emitted = [call for call in range(1, 18) if _next_trace_call(owner, "draft.output")[1]]

    assert object_with_attributes is owner
    assert emitted == [1, 2, 3, 4, 8, 16]


def test_tensor_summary_contains_bounded_values_and_metadata():
    tensor = torch.arange(32, dtype=torch.int32).reshape(4, 8)

    summary = tensor_summary(tensor)

    assert "shape=(4, 8)" in summary
    assert "dtype=torch.int32" in summary
    assert "sample_n=32" in summary
    assert "head=[0.0, 1.0" in summary
    assert "31.0" not in summary.split("head=", 1)[1]


def test_tensor_summary_handles_tensor_lists_without_full_dump():
    summary = tensor_summary([torch.tensor([1, 2]), torch.tensor([3, 4])])

    assert summary.startswith("list[0:shape=(2,)")
    assert "1:shape=(2,)" in summary
