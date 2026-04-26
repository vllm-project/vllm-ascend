import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.spec_decode.draft_model import DraftModelProposer

from vllm_ascend.spec_decode.eagle_proposer import AscendSpecDecodeBaseProposer


class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, False, runner=runner)
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _maybe_share_lm_head(self, target_language_model) -> None:
        # Draft models don't share lm_head with the target model.
        # But we still need to initialize full graph support if enabled.
        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() and self.use_cuda_graph:
            from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

            self.update_stream = torch.npu.Stream()
            self._runnable = ACLGraphWrapper(self._run_merged_draft, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
