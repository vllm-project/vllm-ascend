# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
    build_per_group_layer_attn_metadata,
)


class AscendGemma4Proposer(Gemma4Proposer, AscendSpecDecodeBaseProposer):
    """Gemma4 MTP proposer using Ascend execution and metadata builders."""

    def load_model(self, target_model) -> None:
        super().load_model(target_model)

        # Gemma4 overrides _maybe_share_lm_head(), so the generic Ascend
        # proposer cannot initialize its draft ACLGraph wrapper there. Delay
        # graph initialization until Gemma4 has finished KV-sharing setup.
        if (
            self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
            and self.use_cuda_graph
            and not hasattr(self, "update_stream")
        ):
            self.update_stream = torch.npu.Stream()
            self._runnable = ACLGraphWrapper(
                self._run_merged_draft,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=self.use_eagle,
                enable_enpu=self.enable_enpu,
            )

    def build_per_group_and_layer_attn_metadata(self, common_attn_metadata, draft_index: int = 0):
        per_layer = build_per_group_layer_attn_metadata(
            self.draft_attn_groups,
            common_attn_metadata,
            self._per_group_block_tables,
            common_attn_metadata.num_reqs,
            lambda group_common_metadata, attn_group: attn_group.get_metadata_builder().build_for_drafting(
                group_common_metadata, draft_index
            ),
        )
        per_group = [per_layer[group.layer_names[0]] for group in self.draft_attn_groups]
        return per_group, per_layer
