# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class DeepseekV4Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration
    of a [`DeepseekV3Model`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the DeepSeek-V3.
    e.g. [bzantium/tiny-deepseek-v3](https://huggingface.co/bzantium/tiny-deepseek-v3)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DeepseekV3Model`]
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 61):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 128):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor or routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the LoRA matrices for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the query/key heads that don't use rotary position embeddings.
        n_group (`int`, *optional*, defaults to 8):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 4):
            Number of selected groups for each token (for each token,
            ensuring the selected experts is only within `topk_group`
            groups).
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts, None means dense model.
        first_k_dense_replace (`int`, *optional*, defaults to 3):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import DeepseekV3Model, DeepseekV4Config

    >>> # Initializing a Deepseek-V3 style configuration
    >>> configuration = DeepseekV4Config()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {  # TODO: only replicate attention layers when > first_k_dense_replace
        "layers.*.mlp.experts.*.gate_proj": "local_colwise",
        "layers.*.mlp.experts.*.up_proj": "local_colwise",
        "layers.*.mlp.experts.*.down_proj": "local_rowwise",
        "layers.*.mlp.experts.*":
        "local",  # each expert is wrapped in a module list
        "layers.*.mlp.shared_experts.gate_proj": "local_colwise",
        "layers.*.mlp.shared_experts.up_proj": "local_colwise",
        "layers.*.mlp.shared_experts.down_proj": "local_rowwise",
        "layers.*.mlp.shared_experts": "local",
        "layers.*.mlp.gate_proj": "local_colwise",
        "layers.*.mlp.up_proj": "local_colwise",
        "layers.*.mlp.down_proj": "local_rowwise",
        "layers.*.mlp":
        "gather",  # This is the only moment where results are gathered
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        if getattr(self, "rope_scaling", None) is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        if getattr(self, "rope_scaling", None) is not None:
            for key in ["beta_fast", "beta_slow", "factor"]:
                if key in self.rope_scaling:
                    self.rope_scaling[key] = float(self.rope_scaling[key])

        rope_config_validation(self)
        self.kv_lora_rank = kwargs.get("o_lora_rank", 1024)
