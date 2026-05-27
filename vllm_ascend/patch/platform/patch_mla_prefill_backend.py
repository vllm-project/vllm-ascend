# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# PR vllm-project/vllm#32623 introduced a new MLAPrefillBackend abstraction.
# When MLAAttention.__init__ calls get_mla_prefill_backend(), the upstream
# selector sees that Ascend NPU returns None for get_device_capability() and
# falls back to FlashAttnPrefillBackend, which asserts flash_attn_varlen_func
# is available — crashing on Ascend.
#
# Ascend's AscendSFAImpl/AscendMLAImpl handles the full forward pass (including
# prefill) via impl.forward(), so prefill_backend.run_prefill_* is never called.
# We register a no-op AscendMLAPrefillBackend and patch get_mla_prefill_backend
# so that MLAAttention.__init__ completes without error.

import torch
import vllm.model_executor.layers.attention.mla_attention

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.20.2"):
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

    class AscendMLAPrefillBackend(MLAPrefillBackend):
        @staticmethod
        def get_name() -> str:
            return "ASCEND"

        @classmethod
        def is_available(cls) -> bool:
            return True

        def run_prefill_new_tokens(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            return_softmax_lse: bool,
        ) -> torch.Tensor:
            raise NotImplementedError("Ascend MLA prefill is handled by AscendSFAImpl/AscendMLAImpl")

        def run_prefill_context_chunk(
            self,
            chunk_idx: int,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            raise NotImplementedError("Ascend MLA prefill is handled by AscendSFAImpl/AscendMLAImpl")

    vllm.model_executor.layers.attention.mla_attention.get_mla_prefill_backend = (
        lambda vllm_config: AscendMLAPrefillBackend
    )
