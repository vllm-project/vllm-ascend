import torch
from vllm.attention.layer import Attention
from vllm.config import (VllmConfig, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import (
    process_weights_after_loading, set_default_torch_dtype)
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.models.deepseek_mtp import CustomDeepSeekMTP


class MtpProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        runner,
    ):
        self.vllm_config = vllm_config
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens)
        self.block_size = vllm_config.cache_config.block_size
        self.runner = runner

    @staticmethod
    def prepare_inputs(
        # [batch_size + 1]
        cu_target_query_lens: torch.Tensor,
        # [batch_size]
        num_rejected_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cu_target_query_lens: [0, a, a + b, a + b + c]
        # num_rejected_tokens: [n1, n2, n3]
        # num_tokens_per_req: [a - n1, b - n2, c - n3]
        # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        # token_indices: [0, 1, ..., a - n1 - 1,
        #                 a, a + 1, ..., a + b - n2 - 1,
        #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]

        # [0, a, a + b, a + b + c] -> [a, b, c]
        query_len_per_req = (cu_target_query_lens[1:] -
                             cu_target_query_lens[:-1])
        # [a, b, c] -> [a - n1, b - n2, c - n3]
        num_tokens_per_req = query_len_per_req - num_rejected_tokens

        cu_num_tokens = torch.empty_like(cu_target_query_lens)
        torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
        cu_num_tokens[0] = 0

        # FIXME(woosuk): Avoid synchronization.
        num_tokens = cu_num_tokens[-1].item()
        token_indices = torch.empty(
            num_tokens,
            dtype=torch.int32,
            device=cu_num_tokens.device,
        )

        BLOCK_SIZE = 1024
        prepare_input_kernel(
            token_indices,
            cu_target_query_lens,
            cu_num_tokens,
            block_size=BLOCK_SIZE,
        )
        return cu_num_tokens, token_indices

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [num_tokens]
        target_slot_mapping: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        # [batch_size + 1] starting with 0
        cu_num_tokens: torch.Tensor,
        # [batch_size, max_num_blocks_per_req]
        block_table: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = cu_num_tokens[1:] - 1

        input_ids = torch.empty_like(target_token_ids)
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        input_ids[:-1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        input_ids[last_token_indices] = next_token_ids

        query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
        max_query_len = query_lens.max().item()

        # FIXME: reorder_batch() needs to be called before build()
        # because fields of attn_metadata_builder needs to be updated.
        # However, currently reorder_batch() takes input_batch and
        # scheduler_output as arguments, we should probably refactor
        # the method to use new data structures which are independent
        # from input_batch and scheduler_output.
        # self.runner.attn_metadata_builder.reorder_batch(
        #     input_batch=self.runner.input_batch,
        #     scheduler_output=self.runner.scheduler_output,
        # )

        attn_metadata = self.runner.attn_metadata_builder.build(
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            query_start_loc=cu_num_tokens,
        )

        with set_ascend_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=target_positions,
                previous_hidden_states=target_hidden_states,
            )
        sample_hidden_states = hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)
        draft_token_ids = logits.argmax(dim=-1)

        # [batch_size, 1]
        return draft_token_ids.view(-1, 1)

    def load_model(self) -> None:
        loader = get_model_loader(self.vllm_config.load_config)

        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_device = self.vllm_config.device_config.device

        with set_default_torch_dtype(
                draft_model_config.dtype), set_current_vllm_config(
                    self.vllm_config):
            self.model = CustomDeepSeekMTP(
                vllm_config=self.vllm_config).to(target_device)

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)

        assert len(draft_attn_layer_names) == 1
        self.attn_layer_name = next(iter(draft_attn_layer_names))

        self.model.load_weights(
            loader.get_all_weights(
                self.vllm_config.speculative_config.draft_model_config,
                self.model))
        process_weights_after_loading(self.model, draft_model_config,
                                      target_device)


# TODO Using torch instead of triton may result in poor performance
def prepare_input_kernel(out_ptr: torch.Tensor, cu_query_lens: torch.Tensor,
                         cu_num_tokens: torch.Tensor, block_size: int):
    device = cu_query_lens.device
    dtype = out_ptr.dtype

    offsets = torch.arange(block_size, device=device, dtype=dtype)
    start_pos = cu_num_tokens[:-1]
    end_pos = cu_num_tokens[1:]
    num_tokens = end_pos - start_pos

    global_indices = (start_pos.view(-1, 1) + offsets.view(1, -1))
    values = (cu_query_lens[:-1].view(-1, 1) + offsets.view(1, -1))

    mask = (offsets.view(1, -1) < num_tokens.view(-1, 1))

    global_indices_flat = global_indices[mask]
    values_flat = values[mask]
    out_ptr[global_indices_flat] = values_flat
