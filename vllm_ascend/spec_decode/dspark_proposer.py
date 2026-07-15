from copy import copy
from dataclasses import replace

from vllm.config import CompilationMode, VllmConfig
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark-specific graph configuration on top of the DFlash proposer."""

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_vllm_config = super()._create_draft_vllm_config()
        if not self.use_cuda_graph:
            return draft_vllm_config

        draft_compilation_config = copy(draft_vllm_config.compilation_config)
        draft_compilation_config.mode = CompilationMode.NONE
        return replace(
            draft_vllm_config,
            compilation_config=draft_compilation_config,
        )

    def _stabilize_padded_graph_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        actual_num_reqs: int,
        padded_num_reqs: int,
    ) -> None:
        if padded_num_reqs <= actual_num_reqs:
            return

        common_attn_metadata.seq_lens[actual_num_reqs:padded_num_reqs].fill_(1)
        for attr_name in ("_seq_lens_cpu", "seq_lens_cpu"):
            seq_lens_cpu = getattr(common_attn_metadata, attr_name, None)
            if seq_lens_cpu is None:
                continue
            seq_lens_cpu = self._adjust_tensor(seq_lens_cpu, padded_num_reqs)
            seq_lens_cpu[actual_num_reqs:padded_num_reqs].fill_(1)
            setattr(common_attn_metadata, attr_name, seq_lens_cpu)

        if hasattr(common_attn_metadata, "actual_seq_lengths_q"):
            query_len = 1 + self.num_speculative_tokens
            common_attn_metadata.actual_seq_lengths_q = [query_len] * padded_num_reqs
