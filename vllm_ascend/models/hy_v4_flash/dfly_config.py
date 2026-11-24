# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy, deepcopy

from vllm.models.hy_v4_flash.configs import SPARSE_ATTENTION
from vllm.v1.worker.gpu.spec_decode.speculator_config import DraftModelSpeculatorConfig

METHOD = "dfly"


class FlashDFlyConfig(DraftModelSpeculatorConfig):
    base_method = "draft_model"
    speculator_class = "vllm_ascend.models.hy_v4_flash.dfly_speculator.FlashDFlySpeculator"
    uses_target_hidden_states = True
    shares_target_positions = False

    @classmethod
    def default_model(cls, spec):
        return spec.target_model_config.model

    @classmethod
    def configure(cls, spec):
        target = spec.target_model_config
        if target.hf_text_config.model_type != "hy_v4_flash" or spec.model != target.model:
            raise ValueError("DFly requires a HYV4 Flash target and its embedded draft checkpoint")
        n = spec.num_speculative_tokens
        if type(n) is not int or not 1 <= n <= 16:
            raise ValueError("DFly requires num_speculative_tokens between 1 and 16")
        tp = spec.target_parallel_config.tensor_parallel_size
        if spec.draft_tensor_parallel_size not in (None, tp):
            raise ValueError("DFly draft and target require the same tensor parallel size")
        if spec.max_model_len not in (None, target.max_model_len):
            raise ValueError("DFly draft reuses the target's context length")
        if spec.quantization not in (None, target.quantization):
            raise ValueError("DFly inherits the target checkpoint quantization")

        root = target.hf_config
        config = deepcopy(target.hf_text_config)
        layers = getattr(root, "dfly_num_hidden_layers", None)
        if type(layers) is not int or layers < 1:
            raise ValueError("DFly requires a positive dfly_num_hidden_layers")
        for field in ("target_index_layer", "target_kv_layers"):
            indices = deepcopy(getattr(root, f"dfly_{field}", None))
            if field == "target_index_layer" and type(indices) is int:
                indices = [indices] * layers
            if (
                not isinstance(indices, list)
                or len(indices) != layers
                or any(
                    type(i) is not int
                    or not 1 <= i < config.num_hidden_layers
                    or config.layer_types[i] != SPARSE_ATTENTION
                    for i in indices
                )
            ):
                raise ValueError(f"DFly {field} must select one sparse target layer per draft layer")
            setattr(config, field, indices)
        if len(set(config.target_kv_layers)) != layers:
            raise ValueError("DFly requires distinct target KV layers")
        ropes = [config.rope_for_layer(i) for i in config.target_index_layer + config.target_kv_layers]
        if any(rope != ropes[0] for rope in ropes):
            raise ValueError("DFly target layers must use the same RoPE")
        if (
            getattr(root, "dfly_mtp_fusion", "slot0") != "slot0"
            or getattr(root, "dfly_enable_hidden_correction", True) is not True
            or getattr(root, "dfly_enable_ihc", False) is not False
            or getattr(root, "dfly_mlp_type", "dense") != "dense"
            or getattr(root, "dfly_attn_causal", True) is not True
        ):
            raise ValueError("DFly requires slot0 fusion, hidden correction, dense FFNs and causal MLA")
        config.mask_token_id = getattr(root, "dfly_mask_token_id", None)
        if type(config.mask_token_id) is not int or not 0 <= config.mask_token_id < config.vocab_size:
            raise ValueError("DFly requires a valid dfly_mask_token_id")
        window = getattr(root, "dfly_context_local_window", None)
        config.context_local_window = 0 if window is None else window
        config.target_index_topk = getattr(root, "dfly_target_index_topk", config.index_topk)
        if (
            type(config.context_local_window) is not int
            or config.context_local_window < 0
            or config.target_index_topk != config.index_topk
        ):
            raise ValueError("DFly requires the target top-k and a nonnegative local window")
        config.num_hidden_layers = layers
        config.layer_types = [SPARSE_ATTENTION] * layers
        config.enable_ihc = False
        config.engram_layer_ids = None
        config.dfly_intermediate_size = getattr(root, "dfly_intermediate_size", None)
        config.hidden_correction_intermediate_size = getattr(
            root, "dfly_hidden_correction_intermediate_size", config.hidden_size
        )
        for width in (config.dfly_intermediate_size, config.hidden_correction_intermediate_size):
            if width is not None and (type(width) is not int or width < 1 or width % tp):
                raise ValueError("DFly FFN widths must be positive and divisible by TP")
        config.dfly_attn_post_proj_norm = getattr(root, "dfly_attn_post_proj_norm", None)
        if config.dfly_attn_post_proj_norm is not None and type(config.dfly_attn_post_proj_norm) is not bool:
            raise ValueError("DFly attention post-projection norm must be boolean")
        config.rope_parameters = {SPARSE_ATTENTION: deepcopy(ropes[0])}

        draft = copy(target)
        draft.hf_config = draft.hf_text_config = config
        spec.draft_model_config = draft
        spec.draft_parallel_config = spec.create_draft_parallel_config(
            spec.target_parallel_config, tp, draft_model_config=None
        )
        spec.draft_parallel_config.enable_expert_parallel = False
        spec.draft_tensor_parallel_size = tp
        spec.prompt_lookup_min = spec.prompt_lookup_max = 0
        # The generic parallel_drafting flag selects the EAGLE input path.
        # DFly constructs its own block of draft queries in propose().
        spec.parallel_drafting = False

    @classmethod
    def apply_options(cls, spec):
        if (
            spec.enable_adaptive_verification
            or spec.draft_sample_method != "greedy"
            or spec.rejection_sample_method != "standard"
            or spec.use_heterogeneous_vocab
            or spec.use_local_argmax_reduction
        ):
            raise ValueError("DFly requires greedy shared-vocabulary drafts and standard verification")

    @property
    def num_lookahead_tokens(self):
        return self.num_speculative_tokens + 1

    def max_num_draft_seqs(self, max_num_batched_tokens):
        return max_num_batched_tokens // self.num_lookahead_tokens
