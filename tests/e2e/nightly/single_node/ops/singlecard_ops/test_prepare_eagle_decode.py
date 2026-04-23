import gc
import pytest
import torch
from vllm.triton_utils import triton

from vllm_ascend.worker.v2.spec_decode.eagle.speculator import  prepare_eagle_decode as prepare_eagle_decode_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


class InputBuffers:
    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)


def prepare_eagle_decode_golden(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    last_token_indices: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    input_hidden_states: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
) -> None:
    num_reqs = draft_tokens.shape[0]

    input_buffers.input_ids[:num_reqs] = draft_tokens

    for i in range(num_reqs):
        src_idx = last_token_indices[i].item()
        input_hidden_states[i] = output_hidden_states[src_idx]

    input_buffers.positions[:num_reqs] = torch.minimum(
        input_buffers.positions[:num_reqs] + 1,
        torch.tensor(max_model_len - 1, dtype=input_buffers.positions.dtype, device=input_buffers.positions.device)
    )

    seq_lens = target_seq_lens[:num_reqs] - num_rejected[:num_reqs]
    seq_lens = torch.minimum(seq_lens + 1, torch.tensor(max_model_len, dtype=seq_lens.dtype, device=seq_lens.device))
    input_buffers.seq_lens[:num_reqs] = seq_lens
    input_buffers.seq_lens[num_reqs:] = 0

    query_start_loc = torch.arange(max_num_reqs + 1, dtype=torch.int32, device=input_buffers.query_start_loc.device)
    query_start_loc = torch.minimum(query_start_loc, torch.tensor(num_reqs, dtype=torch.int32, device=query_start_loc.device))
    input_buffers.query_start_loc[:] = query_start_loc


@pytest.mark.parametrize("num_reqs", [1, 7, 32, 128, 2048])
def test_prepare_eagle_decode(num_reqs):
    init_device_properties_triton()
    torch.manual_seed(0)

    max_num_tokens = 2048
    hidden_size = 4096
    max_num_reqs = 2048
    num_tokens = 1024
    max_model_len = 8192
    device = "npu"

    draft_tokens = torch.randint(
        0, 32000, (num_reqs, ), dtype=torch.int64, device=device
    )
    output_hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.float16, device=device
    )
    last_token_indices = torch.randint(
        0, num_tokens, (num_reqs, ), dtype=torch.int64, device=device
    )
    target_seq_lens = torch.randint(
        1, max_model_len, (num_reqs, ), dtype=torch.int64, device=device
    )
    num_rejected = torch.min(
        torch.randint(1, max_model_len, (num_reqs, ), dtype=torch.int64, device=device),
        target_seq_lens
    )

    input_buffers_triton = InputBuffers(
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        device=device,
    )
    input_hidden_states_triton = torch.zeros(
        max_num_tokens, hidden_size, dtype=torch.float16, device=device
    )
    prepare_eagle_decode_triton(
        draft_tokens,
        output_hidden_states,
        last_token_indices,
        target_seq_lens,
        num_rejected,
        input_buffers_triton,
        input_hidden_states_triton,
        max_model_len,
        max_num_reqs,
    )

    input_buffers_golden = InputBuffers(
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        device=device,
    )
    input_hidden_states_golden = torch.zeros(
        max_num_tokens, hidden_size, dtype=torch.float16, device=device
    )
    prepare_eagle_decode_golden(
        draft_tokens,
        output_hidden_states,
        last_token_indices,
        target_seq_lens,
        num_rejected,
        input_buffers_golden,
        input_hidden_states_golden,
        max_model_len,
        max_num_reqs,
    )

    torch.testing.assert_close(input_buffers_triton.input_ids, input_buffers_golden.input_ids)
    torch.testing.assert_close(input_buffers_triton.positions, input_buffers_golden.positions)
    torch.testing.assert_close(input_buffers_triton.seq_lens, input_buffers_golden.seq_lens)
    torch.testing.assert_close(input_buffers_triton.query_start_loc, input_buffers_golden.query_start_loc)
    torch.testing.assert_close(input_hidden_states_triton, input_hidden_states_golden)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
