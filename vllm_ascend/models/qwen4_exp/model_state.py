# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend V2 model-runner state for Qwen4Exp."""

from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState

from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
    AscendMambaHybridModelState,
)

from .qsa_target_metadata import target_metadata_config_supported, target_metadata_preparation


class AscendQwen4ExpModelState(Qwen4ExpModelState, AscendMambaHybridModelState):
    """Qwen4Exp state with Ascend attention metadata construction.

    Method resolution order keeps the upstream PLE n-gram buffers and input
    preparation from :class:`Qwen4ExpModelState`, while ``prepare_attn``
    resolves to :class:`AscendMambaHybridModelState`, which builds Ascend
    attention metadata and stores it on the state for FULL aclgraph replay
    (``update_full_graph_params`` reads ``model_state.attn_metadata``).
    """

    # This experiment keeps the upstream builder class unchanged. The producer
    # routing scope belongs only to the target state; MTP keeps its existing call path.
    @property
    def _qsa_target_metadata_enabled(self):
        return target_metadata_config_supported(self)

    def prepare_attn(self, input_batch, cudagraph_mode, *args, **kwargs):
        if not self._qsa_target_metadata_enabled:
            return super().prepare_attn(input_batch, cudagraph_mode, *args, **kwargs)
        with target_metadata_preparation(self, input_batch, cudagraph_mode, kwargs.get("for_capture", False)):
            return super().prepare_attn(input_batch, cudagraph_mode, *args, **kwargs)


__all__ = ["AscendQwen4ExpModelState"]
