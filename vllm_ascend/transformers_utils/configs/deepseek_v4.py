# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class DeepseekV4Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV3Model`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-V3.
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
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
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
        # base
        vocab_size=129280,
        hidden_size=4096,
        moe_inter_dim=2048,
        num_hidden_layers=43,
        moe_layer_freq=1,
        n_hash_layers=3,
        num_attention_heads=64,
        # moe
        n_routed_experts=256,
        n_shared_experts=1,
        n_activated_experts=6,
        num_experts_per_tok=6,
        first_k_dense_replace=0,
        score_func="sqrtsoftplus",
        topk_method="noaux_tc",
        routed_scaling_factor=1.5,
        # mqa
        q_lora_rank=1024,
        head_dim=512,
        rope_head_dim=64,
        norm_eps: float = 1e-6,
        o_groups=8,
        o_lora_rank=1024,
        window_size=128,
        compress_ratios=[
            1, 1, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
            128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
            128, 4, 128, 4, 128, 4, 128, 4, 128, 4
        ],
        # yarn
        compress_rope_theta=40000,
        # original_seq_len=65536,
        # rope_theta=10000,
        # rope_factor=4,
        # beta_fast=32,
        # beta_slow=1,
        # rope_theta=10000.0,
        # rope_scaling=None,
        max_seq_len=65536,
        rope_theta=10000.0,
        rope_scaling=None,

        # index
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        # hc
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps: float = 1e-6,
        dtype="bfloat16",
        scale_fmt="ue8m0",

        #
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=False,
        norm_topk_prob=True,
        max_position_embeddings=163840,
        **kwargs,
    ):
        # base
        self.vocab_size = vocab_size
        self.moe_inter_dim = moe_inter_dim
        self.n_hash_layers = n_hash_layers
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.moe_layer_freq = moe_layer_freq
        self.num_attention_heads = num_attention_heads

        # moe
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.score_func = score_func
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.topk_method = topk_method
        self.routed_scaling_factor = routed_scaling_factor

        # mqa
        self.q_lora_rank = q_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.norm_eps = norm_eps
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.window_size = window_size
        self.compress_ratios = [
            1, 1, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
            128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
            128, 4, 128, 4, 128, 4, 128, 4, 128, 4
        ]
        # NOTE: This is only for making is_deepseek_mla is True
        self.kv_lora_rank = o_lora_rank

        # index
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        # hc
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.dtype = dtype
        self.scale_fmt = scale_fmt

        #
        self.moe_intermediate_size = moe_inter_dim
        self.rms_norm_eps = 1e-6
        self.pad_token_id = None
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.tie_word_embeddings = False
        self.attention_bias = False
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = 0.02
        self.hidden_act = 'silu'
        self.norm_topk_prob = norm_topk_prob

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.compress_rope_theta = compress_rope_theta
        self.max_seq_len = max_seq_len
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        if self.rope_scaling is not None:
            for key in ["beta_fast", "beta_slow", "factor"]:
                if key in self.rope_scaling:
                    self.rope_scaling[key] = float(self.rope_scaling[key])

        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
