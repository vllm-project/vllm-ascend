from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union

import vllm.envs as envs
from transformers import PretrainedConfig
from vllm.config import ModelConfig, SpeculativeConfig

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    ConfigType = type[DataclassInstance]
else:
    ConfigType = type

ConfigT = TypeVar("ConfigT", bound=ConfigType)

TaskOption = Literal["auto", "generate", "embedding", "embed", "classify",
                     "score", "reward", "transcription"]

RunnerType = Literal["generate", "pooling", "draft", "transcription"]

HfOverrides = Union[dict[str, Any], Callable[[PretrainedConfig],
                                             PretrainedConfig]]


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
                    == "deepseek_v3" or
                self.target_model_config.hf_text_config.model_type \
                    == "mimo"):
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
            self.draft_model_config = ModelConfig(
                model=self.model,
                task="draft",
                tokenizer=self.target_model_config.tokenizer,
                tokenizer_mode=self.target_model_config.tokenizer_mode,
                trust_remote_code=self.target_model_config.trust_remote_code,
                allowed_local_media_path=self.target_model_config.
                allowed_local_media_path,
                dtype=self.target_model_config.dtype,
                seed=self.target_model_config.seed,
                revision=self.revision,
                code_revision=self.code_revision,
                tokenizer_revision=self.target_model_config.tokenizer_revision,
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
            elif self.draft_model_config.hf_config.model_type == "deepseek_mtp":
                self.method = 'mtp'
            else:
                self.method = "draft_model"

            # Replace hf_config for EAGLE draft_model
            if self.method in ("eagle", "eagle3"):
                if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                    raise ValueError(
                        "Chunked prefill and EAGLE are not compatible "
                        "when using V0.")

                from vllm.platforms import current_platform
                from vllm.transformers_utils.configs.eagle import EAGLEConfig
                if isinstance(self.draft_model_config.hf_config,
                              EAGLEConfig) or current_platform.is_neuron():
                    pass
                else:
                    eagle_config = EAGLEConfig(
                        self.draft_model_config.hf_config, method=self.method)
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


SpeculativeConfig.__post_init__ = __post_init__