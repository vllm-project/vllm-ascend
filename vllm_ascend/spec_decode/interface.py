import enum
from typing import Optional

import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class SpecDcodeType(enum.Enum):
    NGRAM = 0
    EAGLE = 1
    EAGLE3 = 2
    MTP = 4


class Proposer:

    def __init__(self,
                 vllm_config: VllmConfig,
                 device: torch.device = None,
                 runner=None):
        pass

    def load_model(self, model):
        """Called by load_model in model_runner"""
        raise NotImplementedError

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens: int,
                  with_prefill: bool = False,
                  skip_attn: bool = False,
                  num_reqs: int = 0,
                  num_tokens_across_dp: Optional[torch.Tensor] = None):
        """Called by dummy_run in modle_runner"""
        raise NotImplementedError

    def generate_token_ids(self,
                           valid_sampled_token_ids: list[list[int]],
                           sampling_metadata: SamplingMetadata = None,
                           scheduler_output: SchedulerOutput = None,
                           spec_decode_metadata: SpecDecodeMetadata = None,
                           positions: torch.Tensor = None,
                           num_scheduled_tokens: int = 0,
                           hidden_states: torch.Tensor = None,
                           attn_metadata=None,
                           aux_hidden_states: torch.Tensor = None):
        """Called by execute_model in model_runner"""
        raise NotImplementedError
