import torch
import vllm.envs as envs_vllm
from vllm.attention.layer import Attention
from vllm.config import (VllmConfig, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import (
    process_weights_after_loading, set_default_torch_dtype)
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import \
    AscendCommonAttentionMetadata as CommonAttentionMetadata
from vllm_ascend.models.deepseek_mtp import CustomDeepSeekMTP
from vllm_ascend.utils import ProfileExecuteDuration


# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    is_greedy = sampling_metadata.temperature == -1
    temperature = torch.where(is_greedy, 1.0, sampling_metadata.temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = torch.empty_like(probs)
    q.exponential_()
    next_token_ids = probs.div_(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(
            is_greedy,
            greedy_token_ids,
            next_token_ids,
        )
    return next_token_ids, probs


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
        # persistent buffers for graph
        self.input_ids = torch.zeros(self.runner.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.runner.device)
        self.positions = torch.zeros(self.runner.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.runner.device)
        self.hidden_states = torch.zeros(
            (self.runner.max_num_tokens, self.runner.hidden_size),
            dtype=self.runner.dtype,
            device=self.runner.device)
        self.is_mtp_torchair_ready = False

    @staticmethod
    def prepare_inputs(
            # [batch_size + 1]
            cu_target_query_lens: torch.Tensor,
            # [batch_size]
            num_rejected_tokens: torch.Tensor,
            force_one_token: bool = False
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
        if force_one_token:
            # enable force_one_token means we only focus on the last token position of each request
            # token_indices: [batch_size]
            cu_num_tokens = torch.arange(cu_target_query_lens.size(0),
                                         device=cu_target_query_lens.device,
                                         dtype=torch.int32)
            relative_index = query_len_per_req - num_rejected_tokens - 1
            token_indices = cu_target_query_lens[:-1] + relative_index
        else:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = cu_num_tokens[1:] - 1

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
        max_query_len = query_lens.max().item()

        seq_lens = (target_positions[last_token_indices] + 1)

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=cu_num_tokens, seq_lens=seq_lens)

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
        extra_builder_kwargs = self.runner.extra_builder_kwargs

        is_running_torchair = self.runner.torchair_graph_enabled and \
            not self.runner.with_prefill and self.is_mtp_torchair_ready

        if is_running_torchair:
            if num_tokens == 1:
                self.runner.attn_state = AscendAttentionState.DecodeOnly
            num_reqs_pad_size = self.runner.num_reqs_pad_size
            extra_builder_kwargs['num_reqs_pad_size'] = num_reqs_pad_size
            # Assume num token per request is one
            extra_builder_kwargs['num_token_pad_size'] = num_reqs_pad_size
            num_input_tokens = self.runner.num_reqs_pad_size
        else:
            extra_builder_kwargs['num_token_pad_size'] = -1
            extra_builder_kwargs['num_reqs_pad_size'] = 0
            num_input_tokens = num_tokens

        attn_metadata = self.runner.attn_metadata_builder.build(
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            is_mtp_model=True,
            **extra_builder_kwargs)

        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states

        # Assuming force_one_token is on, so each perfill request query_lens is 1
        if attn_metadata.prefill is not None:
            attn_metadata.prefill.query_lens[:] = 1

        with set_ascend_forward_context(attn_metadata,
                                        self.vllm_config,
                                        num_tokens=num_input_tokens):
            with ProfileExecuteDuration().capture_async('mtp_forward'):
                model_kwargs = {}
                model_kwargs["attn_metadata"] = attn_metadata
                if self.runner.torchair_graph_enabled:
                    model_kwargs["kv_caches"] = self.runner.kv_caches[-1:]
                if is_running_torchair:
                    torch._dynamo.mark_static(self.input_ids)
                    torch._dynamo.mark_static(self.positions)
                    torch._dynamo.mark_static(attn_metadata.decode.block_table)
                    torch._dynamo.mark_static(
                        attn_metadata.decode.input_positions)
                    torch._dynamo.mark_static(attn_metadata.slot_mapping)
                    torch._dynamo.mark_static(attn_metadata.decode.attn_mask)
                    for kv in self.runner.kv_caches:
                        assert isinstance(kv,
                                          tuple), "kv_cache must be a tuple"
                        torch._dynamo.mark_static(kv[0])
                        torch._dynamo.mark_static(kv[1])
                    hidden_states = self.torchair_compiled_model(
                        input_ids=self.input_ids[:num_input_tokens],
                        positions=self.positions[:num_input_tokens],
                        previous_hidden_states=self.
                        hidden_states[:num_input_tokens],
                        inputs_embeds=None,
                        **model_kwargs)
                else:
                    hidden_states = self.model(
                        input_ids=self.input_ids[:num_input_tokens],
                        positions=self.positions[:num_input_tokens],
                        previous_hidden_states=self.
                        hidden_states[:num_input_tokens],
                        attn_metadata=attn_metadata,
                        kv_caches=self.runner.kv_caches[-1:])
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
        if self.runner.torchair_graph_enabled and self.is_mtp_torchair_ready:
            import torchair  # type: ignore
            from torchair import patch_for_hcom  # type: ignore

            patch_for_hcom()
            config = torchair.CompilerConfig()
            config.experimental_config.frozen_parameter = True
            config.experimental_config.tiling_schedule_optimize = True
            torch.npu.set_compile_mode(jit_compile=False)
            if not self.runner.use_cached_npu_graph:
                npu_backend = torchair.get_npu_backend(compiler_config=config)
                self.torchair_compiled_model = torch.compile(
                    self.model,
                    dynamic=True,
                    fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    backend=npu_backend)
            else:
                self.torchair_compiled_model = torchair.inference.cache_compile(
                    self.model.forward,
                    dynamic=True,
                    fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    config=config,
                    ge_cache=False)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
    ) -> None:
        attn_metadata = self.runner.attn_metadata_builder.build_torchair_graph_dummy(
            num_reqs=num_tokens, num_actual_tokens=1, is_mtp_model=True)
        with set_ascend_forward_context(None,
                                        self.vllm_config,
                                        num_tokens=num_tokens):
            self.model(input_ids=self.input_ids[:num_tokens],
                       positions=self.positions[:num_tokens],
                       previous_hidden_states=self.hidden_states[:num_tokens],
                       attn_metadata=attn_metadata)


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
