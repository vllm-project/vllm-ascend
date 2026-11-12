# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""310P RC GDN metadata builder.

Share request grouping and graph materialization, without building outputs
that are only consumed by the common GDN/KDA kernels.
"""

from __future__ import annotations

from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)


class GDNAttentionMetadataBuilder310(AscendGDNAttentionMetadataBuilder):
    """310P overrides on top of :class:`AscendGDNAttentionMetadataBuilder`.

    310P consumes top-level non-spec fields and derives recurrent lengths in
    its kernel wrapper; it still uses the shared speculative conv metadata.
    """

    # The 310P conv kernel expects PAD_SLOT_ID, not a valid state block.
    _SPEC_GRAPH_PAD_SLOT_ID = PAD_SLOT_ID
    _USE_COMMON_KERNEL_METADATA = False

    def _can_pad_spec_decode(self, graph_request_count: int, num_spec_decode_tokens: int) -> bool:
        # MTP token buffers can hold (1 + K) tokens per request. Do not apply
        # the common single-token limit to 310P concurrent spec replay.
        return (
            graph_request_count <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.spec_token_indx.numel()
        )


# Keep the name introduced by the 310P ACL graph padding patch so existing
# imports and tests from that patch continue to work after rebasing onto
# upstream/main, whose class name is GDNAttentionMetadataBuilder310.
AscendGDNAttentionMetadataBuilder310 = GDNAttentionMetadataBuilder310


class AscendGDNAttentionBackend310(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder310]:
        return AscendGDNAttentionMetadataBuilder310
