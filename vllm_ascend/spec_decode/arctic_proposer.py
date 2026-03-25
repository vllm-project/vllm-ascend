import numpy as np
import torch
from arctic_inference.envs import ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.patch.platform.patch_arctic_speculator import (
    ArcticLSTMSpeculator,
    ArcticMLPSpeculator,
    padding_size_func,
)


class AscendArcticProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config

        self.model = None
        self.device = None

        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_size = int(vllm_config.speculative_config.draft_model_config.hf_text_config.input_hidden_dim)
        self.dtype = vllm_config.model_config.dtype

    def load_model(
        self,
        model: ArcticMLPSpeculator | ArcticLSTMSpeculator,
    ):
        from vllm.config import VllmConfig

        draft_config_model_config = self.speculative_config.draft_model_config

        spec_model_archs = draft_config_model_config.hf_config.architectures
        if not isinstance(spec_model_archs, list):
            logger.error(f"Draft model architectures {spec_model_archs} is not a list. ")
            raise TypeError()
        if len(spec_model_archs) != 1:
            logger.error(f"Draft model architectures {spec_model_archs} does not have exactly one architecture. ")
            raise ValueError()
        if spec_model_archs[0] not in [
            "ArcticMLPSpeculatorPreTrainedModel",
            "ArcticLSTMSpeculatorPreTrainedModel",
            "MLPVariantSpeculatorPreTrainedModel",
        ]:
            logger.error(f"Draft model architecture {spec_model_archs} is not supported by Arctic Speculator. ")
            raise ValueError()

        draft_config_model_config.hf_config.update(
            {
                "num_hidden_layers": 0,
            }
        )

        if not ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK:
            base_model_arch = self.vllm_config.model_config.architectures[0]
            if not hasattr(draft_config_model_config.hf_config, "base_model_archs"):
                raise ValueError(
                    "Draft model config does not have base_model_archs attribute. "
                    "Set ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK=1 to skip this assertion."
                )
            base_model_archs_in_spec_config = draft_config_model_config.hf_config.base_model_archs
            if base_model_arch not in base_model_archs_in_spec_config:
                raise ValueError(
                    f"Draft model trained with base model architectures {base_model_archs_in_spec_config} "
                    f"does not match the base model architecture {base_model_arch} in the vLLM config. "
                    "Set ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK=1 to skip this assertion."
                )

        draft_config_quant_config = VllmConfig._get_quantization_config(
            self.vllm_config.model_config,
            self.vllm_config.load_config,
        )
        self.speculative_config.draft_parallel_config.worker_cls = self.vllm_config.parallel_config.sd_worker_cls
        draft_config_parallel_config = self.speculative_config.draft_parallel_config

        # We cannot use deepcopy here because Ulysses introduces
        # torch._C._distributed_c10d.ProcessGroup objects that are not
        # designed to be pickled.
        draft_worker_config = VllmConfig(
            model_config=draft_config_model_config,
            quant_config=draft_config_quant_config,
            parallel_config=draft_config_parallel_config,
            load_config=self.vllm_config.load_config,
            device_config=self.vllm_config.device_config,
        )
        draft_worker_config.scheduler_config = self.vllm_config.scheduler_config

        self.model = get_model(vllm_config=draft_worker_config)
        self.device = next(self.model.parameters()).device

        self.input_hidden_dim: int
        if isinstance(self.model, ArcticLSTMSpeculator):
            self.input_hidden_dim = self.model.input_hidden_dim
        else:
            self.input_hidden_dim = self.model.emb_dim

    def prepare_hidden_states(
        self,
        sample_hidden_states: torch.Tensor,
        sampled_token_ids: np.ndarray | list[list[int]],
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        if sample_hidden_states is not None:
            assert sample_hidden_states.shape[-1] == self.input_hidden_dim, (
                f"hidden_states shape mismatch: {sample_hidden_states.shape[-1]} != {self.input_hidden_dim}. \
                Please make sure spec model is trained using the same base model."
            )

        if isinstance(sampled_token_ids, np.ndarray):
            max_gen_len = sampled_token_ids.shape[-1]
        else:
            max_gen_len = len(sampled_token_ids[0]) if sampled_token_ids else 0
        if max_gen_len == 1:
            return sample_hidden_states

        assert spec_decode_metadata is not None
        if isinstance(sampled_token_ids, np.ndarray):
            valid_mask = sampled_token_ids != -1
            gen_lens = valid_mask.sum(axis=1)
        else:
            valid_mask = [tok != -1 for tokens in sampled_token_ids for tok in tokens]
            gen_lens = torch.tensor(
                [sum(1 for j, tok in enumerate(tokens) if tok != -1) for tokens in sampled_token_ids],
                device=sample_hidden_states.device if sample_hidden_states is not None else "cpu",
            )
        num_sampled_tokens = np.array(spec_decode_metadata.num_draft_tokens)
        num_sampled_tokens = torch.tensor(num_sampled_tokens, device=gen_lens.device) + 1
        hidden_states_idx = (gen_lens - 1) + torch.cumsum(num_sampled_tokens, 0) - num_sampled_tokens
        previous_hidden_states = sample_hidden_states[hidden_states_idx]

        return previous_hidden_states

    def propose(
        self,
        context_token_ids: np.ndarray,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> np.ndarray:
        assert num_predict_tokens > 0, f"num_predict_tokens must be greater than 0, got {num_predict_tokens}."

        input_ids = torch.tensor(context_token_ids, device=self.device)

        next_tokens = self.model.generate_proposals(
            input_ids=input_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=num_predict_tokens,
        )

        return next_tokens.cpu().numpy()

    def propose_draft_token_ids(
        self,
        context_token_ids: np.ndarray,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> np.ndarray:
        draft_token_ids = self.propose(
            context_token_ids=context_token_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=num_predict_tokens,
        )

        return draft_token_ids

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ) -> None:
        num_predict_tokens = self.vllm_config.speculative_config.num_speculative_tokens
        size = padding_size_func(self.vllm_config.scheduler_config.max_num_seqs)
        input_ids = torch.rand(size, dtype=self.dtype)
        previous_hidden_states = torch.zeros(
            (size, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
        ):
            self.model.generate_proposals(
                input_ids=input_ids,
                previous_hidden_states=previous_hidden_states,
                num_predict_tokens=num_predict_tokens,
            )
