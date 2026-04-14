import torch
from vllm.v1.sample.logits_processor import LogitsProcessors

from vllm_ascend._310p.block_table import MultiGroupBlockTable
from vllm_ascend.worker.npu_input_batch import NPUInputBatch


class NPUInputBatch310(NPUInputBatch):
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],
        kernel_block_sizes: list[list[int]],
        max_num_blocks_per_req: list[int] | None = None,
        logitsprocs: LogitsProcessors | None = None,
        logitsprocs_need_output_token_ids: bool = False,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
        num_speculative_tokens: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ):
        super().__init__(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            device=device,
            pin_memory=pin_memory,
            vocab_size=vocab_size,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            max_num_blocks_per_req=max_num_blocks_per_req,
            logitsprocs=logitsprocs,
            logitsprocs_need_output_token_ids=logitsprocs_need_output_token_ids,
            is_spec_decode=is_spec_decode,
            is_pooling_model=is_pooling_model,
            num_speculative_tokens=num_speculative_tokens,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            max_num_blocks=max_num_blocks_per_req,
            num_speculative_tokens=num_speculative_tokens,
            kernel_sizes=kernel_block_sizes,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
