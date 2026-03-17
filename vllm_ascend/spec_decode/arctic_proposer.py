from typing import Optional, Union

from vllm.config import VllmConfig, CUDAGraphMode
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.logger import init_logger

import numpy as np
import torch

from vllm_ascend.patch.platform.patch_arctic_speculator import (padding_size, 
                                            ArcticMLPSpeculator, ArcticLSTMSpeculator)
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from arctic_inference.envs import ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK

logger = init_logger(__name__)

class AscendArcticProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        super().__init__(vllm_config)

        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config

        self.model = None
        self.device = None

        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_size = int(
            vllm_config.speculative_config.draft_model_config.get_hidden_size()
        )
        self.dtype = vllm_config.model_config.dtype
    
    def load_model(
        self,
        model: Union[ArcticMLPSpeculator, ArcticLSTMSpeculator],
    ):
        from vllm.config import VllmConfig

        draft_config_model_config = self.speculative_config.draft_model_config

        spec_model_archs = draft_config_model_config.hf_config.architectures
        if not isinstance(spec_model_archs, list):
            logger.error(
                f"Draft model architectures {spec_model_archs} is not a list. "
            )
            raise TypeError()
        if len(spec_model_archs) != 1:
            logger.error(
                f"Draft model architectures {spec_model_archs} does not have exactly one architecture. "
            )
            raise ValueError()
        if spec_model_archs[0] not in [
                "ArcticMLPSpeculatorPreTrainedModel",
                "ArcticLSTMSpeculatorPreTrainedModel",
                "MLPVariantSpeculatorPreTrainedModel",
        ]:
            logger.error(
                f"Draft model architecture {spec_model_archs} is not supported by Arctic Speculator. "
            )
            raise ValueError()

        if not ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK:
            base_model_arch = self.vllm_config.model_config.architectures[0]
            if not hasattr(draft_config_model_config.hf_config, "base_model_archs"):
                logger.error(
                    "Draft model config does not have base_model_archs attribute. "
                    "Set ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK=1 to skip this assertion."
                )
                assert False
            base_model_archs_in_spec_config = draft_config_model_config.hf_config.base_model_archs
            if base_model_arch not in base_model_archs_in_spec_config:
                logger.error(
                    f"Draft model trained with base model architectures {base_model_archs_in_spec_config} "
                    f"does not match the base model architecture {base_model_arch} in the vLLM config. "
                    "Set ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK=1 to skip this assertion."
                )
                assert False

        draft_config_quant_config = VllmConfig._get_quantization_config(
            self.vllm_config.model_config,
            self.vllm_config.load_config,
        )
        self.speculative_config.draft_parallel_config.worker_cls =\
            self.vllm_config.parallel_config.sd_worker_cls
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
        self.device = next(model.parameters()).device

        self.input_hidden_dim = self.model.input_hidden_dim if isinstance(
            self.model, ArcticLSTMSpeculator) else self.model.emb_dim

    def prepare_hidden_states(
        self,
        sample_hidden_states: torch.Tensor,
        sampled_token_ids: Union[np.ndarray, list[list[int]]],
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        if sample_hidden_states is not None:
            assert sample_hidden_states.shape[-1] == self.input_hidden_dim, \
                f"hidden_states shape mismatch: {sample_hidden_states.shape[-1]} != {self.input_hidden_dim}. \
                Please make sure spec model is trained using the same base model."
        
        # if isinstance(sampled_token_ids, list):
        #     # Pad the list of lists to create a uniform tensor
        #     max_len = max(len(x) for x in sampled_token_ids) if sampled_token_ids else 0
        #     if max_len == 0:
        #         return sample_hidden_states
        #     padded_ids = [l + [-1] * (max_len - len(l)) for l in sampled_token_ids]
        #     sampled_token_ids = torch.tensor(padded_ids,
        #                                  device=sample_hidden_states.device)

        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            return sample_hidden_states

        assert spec_decode_metadata is not None
        valid_mask = sampled_token_ids != -1
        gen_lens = valid_mask.sum(dim=1)
        num_sampled_tokens = np.array(spec_decode_metadata.num_draft_tokens)
        num_sampled_tokens = torch.tensor(num_sampled_tokens,
                                          device=gen_lens.device) + 1
        hidden_states_idx = (gen_lens - 1) + torch.cumsum(
            num_sampled_tokens, 0) - num_sampled_tokens
        previous_hidden_states = sample_hidden_states[hidden_states_idx]

        return previous_hidden_states

    def propose(
        self,
        context_token_ids: np.ndarray,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> Optional[np.ndarray]:
        assert num_predict_tokens > 0, \
            f"num_predict_tokens must be greater than 0, got {num_predict_tokens}."
        
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
    ) -> Optional[np.ndarray]:
        draft_token_ids = self.propose(
            context_token_ids=context_token_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=num_predict_tokens
            )
        
        return draft_token_ids
    
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False
    ) -> None:
        num_predict_tokens = self.vllm_config.speculative_config.num_speculative_tokens
        size = padding_size(self.vllm_config.scheduler_config.max_num_seqs)
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
                is_draft_model=True):
            self.model.generate_proposals(
                input_ids=input_ids,
                previous_hidden_states=previous_hidden_states,
                num_predict_tokens=num_predict_tokens,
            )

    def propose(
        self,
        context_token_ids: np.ndarray,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> Optional[np.ndarray]:
        assert num_predict_tokens > 0, \
            f"num_predict_tokens must be greater than 0, got {num_predict_tokens}."
        
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
    ) -> Optional[np.ndarray]:
        draft_token_ids = self.propose(
            context_token_ids=context_token_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=num_predict_tokens
            )
        
        return draft_token_ids
    
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False
    ) -> None:
        num_predict_tokens = self.vllm_config.speculative_config.num_speculative_tokens
        size = padding_size(self.vllm_config.scheduler_config.max_num_seqs)
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
                is_draft_model=True):
            self.model.generate_proposals(
                input_ids=input_ids,
                previous_hidden_states=previous_hidden_states,
                num_predict_tokens=num_predict_tokens,
            )
