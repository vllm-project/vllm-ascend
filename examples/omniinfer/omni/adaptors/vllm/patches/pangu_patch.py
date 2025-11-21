from omni.adaptors.vllm.utils import get_attr_by_names


# The following patches are corresponding to vllm-0.9.0
def patch_pangu():
    from vllm.config import ModelConfig

    @property
    def is_deepseek_mla(self) -> bool:
        kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
        kv_lora_dim = get_attr_by_names(self.hf_text_config, kv_lora_dim_names,
                                        None)
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in \
            ('deepseek_v2', 'deepseek_v3', 'deepseek_v32', 'deepseek_mtp', 'pangu_ultra_moe'):
            return kv_lora_dim is not None
        elif self.hf_text_config.model_type == 'eagle':
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return self.hf_text_config.model.model_type in \
                    ('deepseek_v2', 'deepseek_v3', 'deepseek_v32', 'pangu_ultra_moe') \
                and kv_lora_dim is not None
        return False

    def _verify_with_expert_parallelism(self) -> None:
        num_expert_names = [
            "moe_num_experts",  # Dbrx
            "num_experts",  # Jamba
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
            "num_routed_experts",  # Pangu
        ]
        num_experts = 0
        for name in num_expert_names:
            num_experts = getattr(self.hf_text_config, name, 0)
            if num_experts > 0:
                break
        if num_experts < 1:
            raise ValueError(
                "Number of experts in the model must be greater than 0 "
                "when expert parallelism is enabled.")

    def get_head_size(self) -> int:
        if self.is_deepseek_mla:
            qk_rope_dim_names = ['attention_qk_rope_dim', 'qk_rope_head_dim']
            kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
            qk_rope_dim = get_attr_by_names(self.hf_text_config,
                                            qk_rope_dim_names, 0)
            kv_lora_dim = get_attr_by_names(self.hf_text_config,
                                            kv_lora_dim_names, 0)
            if self.use_mla:
                return kv_lora_dim + qk_rope_dim
            else:
                qk_dim_names = ['attention_qk_dim', 'qk_nope_head_dim']
                qk_dim = get_attr_by_names(self.hf_text_config, qk_dim_names,
                                           0)
                if qk_rope_dim and qk_dim:
                    return qk_rope_dim + qk_dim

    # for mtp
    import vllm.envs as envs
    from transformers import PretrainedConfig
    from vllm.config import SpeculativeConfig

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hf_config.model_type in ["deepseek_v3", "deepseek_v32"]:
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "n_predict": n_predict,
                "architectures": ["DeepSeekMTPModel"]
            })

        if hf_config.model_type == "pangu_ultra_moe":
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_mtp_layers", None)
            hf_config.update({
                "n_predict": n_predict,
                "architectures": ["PanguUltraMoEMTPModel"]
            })

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["MiMoMTPModel"]
            })
            return hf_config

        return hf_config

    def __post_init__(self):

        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        if self.model is None and self.num_speculative_tokens is not None:
            # TODO(Shangming): Refactor mtp configuration logic when supporting
            # mtp acceleration for more models besides deepseek_v3
            if self.target_model_config and \
                (self.target_model_config.hf_text_config.model_type \
                        in ["deepseek_v3", "deepseek_v32"] or
                    self.target_model_config.hf_text_config.model_type \
                        == "mimo" or
                    self.target_model_config.hf_text_config.model_type \
                        == "pangu_ultra_moe"):
                # use the draft model from the same model:
                self.model = self.target_model_config.model
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            else:
                raise ValueError("num_speculative_tokens was provided without "
                                 "speculative model.")

        # Automatically configure the method for ngram when "model" is used
        # instead of "method"
        if self.method is None and (self.model is not None
                                    and self.model in ("ngram", "[ngram]")):
            self.method = "ngram"

        if self.method in ("ngram", "[ngram]"):
            # Unified to "ngram" internally
            self.method = "ngram"
            # Set default values if not provided
            if (self.prompt_lookup_min is None
                    and self.prompt_lookup_max is None):
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                assert self.prompt_lookup_max is not None
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                assert self.prompt_lookup_min is not None
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min < 1:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must be > 0")
            if self.prompt_lookup_max < 1:
                raise ValueError(
                    f"prompt_lookup_max={self.prompt_lookup_max} must be > 0")
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}")

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    task="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.
                    trust_remote_code,
                    allowed_local_media_path=self.target_model_config.
                    allowed_local_media_path,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.
                    tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.
                    max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_seq_len_to_capture=self.target_model_config.
                    max_seq_len_to_capture,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                )

                # Automatically detect the method
                if self.method in ('eagle', 'eagle3'):
                    pass
                elif "eagle-" in self.draft_model_config.model.lower() or \
                        "eagle3-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif (self.draft_model_config.hf_config.model_type ==
                      "mlp_speculator"):
                    self.method = "mlp_speculator"
                elif (self.draft_model_config.hf_config.model_type ==
                      "deepseek_mtp"):
                    self.method = "deepseek_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                                "All Deepseek MTP models only have " \
                                "one layer. Might need some code changes " \
                                "to support multiple layers."
                            )
                elif (self.draft_model_config.hf_config.model_type ==
                      "pangu_ultra_moe_mtp"):
                    self.method = "pangu_ultra_moe_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                                "All Pangu Ultra MoE MTP models only have " \
                                "one layer. Might need some code changes " \
                                "to support multiple layers."
                            )
                else:
                    self.method = "draft_model"

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                        raise ValueError(
                            "Chunked prefill and EAGLE are not compatible "
                            "when using V0.")

                    from vllm.transformers_utils.configs.eagle import \
                        EAGLEConfig
                    if isinstance(self.draft_model_config.hf_config,
                                  EAGLEConfig):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method)
                        self.draft_model_config.hf_config = eagle_config

                if (self.num_speculative_tokens is not None
                        and hasattr(self.draft_model_config.hf_config,
                                    "num_lookahead_tokens")):
                    self.draft_model_config.hf_config.num_lookahead_tokens = \
                    self.num_speculative_tokens

                n_predict = getattr(self.draft_model_config.hf_config,
                                    "n_predict", None)
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif self.num_speculative_tokens > n_predict and \
                            self.num_speculative_tokens % n_predict != 0:
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}")

                self.draft_tensor_parallel_size = \
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    ))

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size))

        if self.acceptance_method == "typical_acceptance_sampler":
            if self.posterior_threshold is None:
                self.posterior_threshold = 0.09
            if self.posterior_alpha is None:
                self.posterior_alpha = 0.3

        self._verify_args()

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "deepseek_mtp", "ernie_mtp",
                               "pangu_ultra_moe_mtp")

    ModelConfig.is_deepseek_mla = is_deepseek_mla
    ModelConfig._verify_with_expert_parallelism = _verify_with_expert_parallelism
    ModelConfig.get_head_size = get_head_size

    SpeculativeConfig.__post_init__ = __post_init__
    SpeculativeConfig.hf_config_override = hf_config_override
    SpeculativeConfig.use_eagle = use_eagle

    print("++++++++++++++++++++++patch_pangu++++++++++++++++++++++++++++")
