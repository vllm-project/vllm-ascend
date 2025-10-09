import ast

import vllm.envs as envs
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import logger


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
        if (self.target_model_config
                and self.target_model_config.hf_text_config.model_type
                in ("deepseek_v3", "deepseek_v32", "mimo", "ernie4_5_moe",
                    "qwen3_next")):
            # use the draft model from the same model:
            self.model = self.target_model_config.model
            # Align the quantization of draft model for cases such as
            # --quantization fp8 with a bf16 checkpoint.
            if not self.quantization:
                self.quantization = self.target_model_config.quantization
        elif self.method in ("ngram", "[ngram]"):
            self.model = "ngram"
        else:
            raise ValueError("num_speculative_tokens was provided but without "
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
        if (self.prompt_lookup_min is None and self.prompt_lookup_max is None):
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
            # TODO: Move this import to the top once `ModelConfig`
            # lives in `vllm.config.model`.
            from vllm.config import ModelConfig
            self.draft_model_config = ModelConfig(
                model=self.model,
                runner="draft",
                tokenizer=self.target_model_config.tokenizer,
                tokenizer_mode=self.target_model_config.tokenizer_mode,
                trust_remote_code=self.target_model_config.trust_remote_code,
                allowed_local_media_path=self.target_model_config.
                allowed_local_media_path,
                allowed_media_domains=self.target_model_config.
                allowed_media_domains,
                dtype=self.target_model_config.dtype,
                seed=self.target_model_config.seed,
                revision=self.revision,
                code_revision=self.code_revision,
                tokenizer_revision=self.target_model_config.tokenizer_revision,
                spec_target_max_model_len=self.target_model_config.
                max_model_len,
                quantization=self.quantization,
                enforce_eager=self.target_model_config.enforce_eager,
                max_logprobs=self.target_model_config.max_logprobs,
                hf_overrides=SpeculativeConfig.hf_config_override,
            )

            # Automatically detect the method
            if self.method in ('eagle', 'eagle3'):
                pass
            # examples:
            # yuhuili/EAGLE-LLaMA3-Instruct-8B
            # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
            # AngelSlim/Qwen3-8B_eagle3
            elif "eagle-" in self.draft_model_config.model.lower():
                self.method = "eagle"
            elif "eagle3" in self.draft_model_config.model.lower():
                self.method = "eagle3"
            elif self.draft_model_config.hf_config.model_type == "medusa":
                self.method = "medusa"
            elif (self.draft_model_config.hf_config.model_type ==
                  "mlp_speculator"):
                self.method = "mlp_speculator"
            elif (self.draft_model_config.hf_config.model_type
                  in ("deepseek_mtp", "mimo_mtp", "glm4_moe_mtp")):
                self.method = "deepseek_mtp"
                if self.num_speculative_tokens > 1:
                    logger.warning(
                            "All Deepseek MTP models only have " \
                            "one layer. Might need some code changes " \
                            "to support multiple layers."
                        )
            elif (self.draft_model_config.hf_config.model_type == "ernie_mtp"):
                self.method = "ernie_mtp"
                if self.num_speculative_tokens > 1:
                    logger.warning(
                            "All Ernie MTP models only have " \
                            "one layer. Might need some code changes " \
                            "to support multiple layers."
                        )
            elif (self.draft_model_config.hf_config.model_type ==
                  "qwen3_next_mtp"):
                self.method = "qwen3_next_mtp"
                if self.num_speculative_tokens > 1:
                    logger.warning(
                            "All Qwen3Next MTP models only have " \
                            "one layer. Might need some code changes " \
                            "to support multiple layers."
                        )
            elif (self.draft_model_config.hf_config.model_type
                  in ("longcat_flash_mtp")):
                self.method = "longcat_flash_mtp"
                if self.num_speculative_tokens > 1:
                    logger.warning(
                            "LongCat MTP models only have " \
                            "one layer. Might need some code changes " \
                            "to support multiple layers."
                        )
            else:
                self.method = "draft_model"
                raise NotImplementedError(
                    "Speculative decoding with draft model is not "
                    "supported yet. Please consider using other "
                    "speculative decoding methods such as ngram, medusa, "
                    "eagle, or deepseek_mtp.")

            # Replace hf_config for EAGLE draft_model
            if self.method in ("eagle", "eagle3"):
                if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                    raise ValueError(
                        "Chunked prefill and EAGLE are not compatible "
                        "when using V0.")

                from vllm.transformers_utils.configs import SpeculatorsConfig
                from vllm.transformers_utils.configs.eagle import EAGLEConfig

                if isinstance(self.draft_model_config.hf_config,
                              (EAGLEConfig, SpeculatorsConfig)):
                    pass
                else:
                    eagle_config = EAGLEConfig(
                        self.draft_model_config.hf_config,
                        method=self.method,
                        model_type="eagle")
                    self.draft_model_config.hf_config = eagle_config

            if (self.num_speculative_tokens is not None
                    and hasattr(self.draft_model_config.hf_config,
                                "num_lookahead_tokens")):
                self.draft_model_config.hf_config.num_lookahead_tokens = \
                self.num_speculative_tokens

            n_predict = getattr(self.draft_model_config.hf_config, "n_predict",
                                None)
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

            if self.speculative_token_tree is None:
                # Generate chain of tokens.
                self.speculative_token_tree = str([
                    (i + 1) * (0, ) for i in range(self.num_speculative_tokens)
                ])
            else:
                # Sort the token tree breadth-first.
                tree_choices = ast.literal_eval(self.speculative_token_tree)
                self.speculative_token_tree = str(
                    sorted(tree_choices, key=lambda t: (len(t), t)))

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


SpeculativeConfig.__post_init__ = __post_init__
