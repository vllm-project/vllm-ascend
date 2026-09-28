import gc

import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel_ascend,
)


def test_prepare_dflash_inputs_clamps_seq_len_to_max_model_len():
    init_device_properties_triton()
    device = "npu"
    max_model_len = 16
    num_query_per_req = 3
    num_speculative_steps = 2

    outputs = {
        "out_input_ids_ptr": torch.empty(num_query_per_req, dtype=torch.int32, device=device),
        "out_query_positions_ptr": torch.empty(num_query_per_req, dtype=torch.int32, device=device),
        "out_query_start_loc_ptr": torch.empty(2, dtype=torch.int32, device=device),
        "out_seq_lens_ptr": torch.empty(1, dtype=torch.int32, device=device),
        "out_query_slot_mapping_ptr": torch.empty(num_query_per_req, dtype=torch.int32, device=device),
        "out_context_positions_ptr": torch.empty(1, dtype=torch.int32, device=device),
        "out_context_slot_mapping_ptr": torch.empty(1, dtype=torch.int32, device=device),
        "out_sample_indices_ptr": torch.empty(num_speculative_steps, dtype=torch.int32, device=device),
        "out_sample_pos_ptr": torch.empty(num_speculative_steps, dtype=torch.int32, device=device),
        "out_sample_idx_mapping_ptr": torch.empty(num_speculative_steps, dtype=torch.int32, device=device),
        "out_temperature_ptr": torch.empty(1, dtype=torch.float32, device=device),
        "out_seeds_ptr": torch.empty(1, dtype=torch.int64, device=device),
    }
    inputs = {
        "target_positions_ptr": torch.tensor([max_model_len - 1], dtype=torch.int32, device=device),
        "target_query_start_loc_ptr": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "idx_mapping_ptr": torch.tensor([0], dtype=torch.int32, device=device),
        "last_sampled_ptr": torch.tensor([42], dtype=torch.int32, device=device),
        "next_prefill_tokens_ptr": torch.tensor([43], dtype=torch.int32, device=device),
        "num_sampled_ptr": torch.tensor([1], dtype=torch.int32, device=device),
        "num_rejected_ptr": torch.tensor([0], dtype=torch.int32, device=device),
        "temperature_ptr": torch.tensor([1.0], dtype=torch.float32, device=device),
        "seeds_ptr": torch.tensor([0], dtype=torch.int64, device=device),
        "block_table_ptr": torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device),
    }
    kwargs = {
        **outputs,
        **inputs,
        "block_table_stride": 3,
        "parallel_drafting_token_id": 151643,
        "block_size": 8,
        "num_query_per_req": num_query_per_req,
        "num_speculative_steps": num_speculative_steps,
        "max_num_reqs": 1,
        "max_num_tokens": num_query_per_req,
        "max_model_len": max_model_len,
        "SAMPLE_FROM_ANCHOR": False,
        "PAD_SLOT_ID": -1,
        "BLOCK_SIZE": 1,
    }
    kwargs.update(cp_rank=0, CP_SIZE=1, CP_INTERLEAVE=1)

    _prepare_dflash_inputs_kernel_ascend[(1, 1)](**kwargs)

    torch.testing.assert_close(
        outputs["out_seq_lens_ptr"],
        torch.tensor([max_model_len], dtype=torch.int32, device=device),
    )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
