import pytest
import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401


def _reference(draft_token_ids, target_argmax, bonus_token_ids, counts, max_spec_len):
    output = torch.full((len(counts), max_spec_len + 1), -1, dtype=torch.int32)
    cursor = 0
    for request_index, count in enumerate(counts):
        all_accepted = True
        for position in range(count):
            target_token = int(target_argmax[cursor + position])
            output[request_index, position] = target_token
            if int(draft_token_ids[cursor + position]) != target_token:
                all_accepted = False
                break
        if all_accepted:
            output[request_index, count] = int(bonus_token_ids[request_index])
        cursor += count
    return output


def _make_inputs(batch_size, max_spec_len):
    if batch_size == 1:
        counts = [max_spec_len]
    else:
        counts = [0] + [1 + (index * 3) % max_spec_len for index in range(1, batch_size)]
    total_tokens = sum(counts)
    draft_token_ids = torch.arange(100, 100 + total_tokens, dtype=torch.int32)
    target_argmax = draft_token_ids.to(torch.int64)

    cursor = 0
    for request_index, count in enumerate(counts):
        if count and request_index % 4:
            if request_index % 4 == 1:
                mismatch_position = 0
            elif request_index % 4 == 2:
                mismatch_position = count // 2
            else:
                mismatch_position = count - 1
            target_argmax[cursor + mismatch_position] += 1000
        cursor += count

    cu_num_draft_tokens = torch.tensor(counts, dtype=torch.int32).cumsum(0).to(torch.int32)
    bonus_token_ids = torch.arange(2000, 2000 + batch_size, dtype=torch.int32).reshape(-1, 1)
    return counts, cu_num_draft_tokens, draft_token_ids, target_argmax, bonus_token_ids


@pytest.mark.parametrize(
    "batch_size,max_spec_len",
    [(1, 5), (2, 7), (9, 15), (10, 15)],
)
def test_rejection_sample_greedy_310(batch_size, max_spec_len):
    counts, cu_num_draft_tokens, draft_token_ids, target_argmax, bonus_token_ids = _make_inputs(
        batch_size, max_spec_len
    )
    expected = _reference(
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        counts,
        max_spec_len,
    )
    output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device="npu")

    torch.ops._C_ascend.npu_rejection_sample_greedy_310(
        cu_num_draft_tokens.npu(),
        draft_token_ids.npu(),
        target_argmax.npu(),
        bonus_token_ids.npu(),
        output,
        max_spec_len,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(output.cpu(), expected)


@torch.inference_mode()
def test_rejection_sample_greedy_310_aclgraph_replay():
    batch_size = 10
    max_spec_len = 15
    counts, cu_num_draft_tokens, draft_token_ids, target_argmax, bonus_token_ids = _make_inputs(
        batch_size, max_spec_len
    )
    expected = _reference(
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        counts,
        max_spec_len,
    )

    cu_num_draft_tokens_npu = cu_num_draft_tokens.npu()
    draft_token_ids_npu = draft_token_ids.npu()
    target_argmax_npu = target_argmax.npu()
    bonus_token_ids_npu = bonus_token_ids.npu()
    output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device="npu")

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        torch.ops._C_ascend.npu_rejection_sample_greedy_310(
            cu_num_draft_tokens_npu,
            draft_token_ids_npu,
            target_argmax_npu,
            bonus_token_ids_npu,
            output,
            max_spec_len,
        )
    graph.replay()
    torch_npu.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected)

    updated_target_argmax = target_argmax + 7
    updated_bonus_token_ids = bonus_token_ids + 123
    updated_expected = _reference(
        draft_token_ids,
        updated_target_argmax,
        updated_bonus_token_ids,
        counts,
        max_spec_len,
    )
    target_argmax_npu.copy_(updated_target_argmax)
    bonus_token_ids_npu.copy_(updated_bonus_token_ids)
    output.fill_(42)
    graph.replay()
    torch_npu.npu.synchronize()
    torch.testing.assert_close(output.cpu(), updated_expected)
