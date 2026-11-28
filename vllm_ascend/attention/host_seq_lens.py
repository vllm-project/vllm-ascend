#
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
# This file is a part of the vllm-ascend project.
#
"""Legacy exact host seq-lens backend registry for Model Runner V2.

After a speculative-decoding step the runner corrects the per-request CPU
sequence-length mirror with the device-rejected count (producer D2H copy in
``_copy_num_computed_tokens_to_cpu`` plus ``event.synchronize()`` in
``_update_seq_lens_cpu``). That mirror only exists for legacy attention
backends whose metadata builders feed *host* length lists to CANN kernels:

- FIA/GQA base builder (``attention_v1``): ``seq_lens.tolist()`` /
  ``query_start_loc_cpu[1:].tolist()`` -> ``actual_seq_lengths_{kv,q}``.
- MLA builder (``mla_v1``): host ``seq_lens_cpu`` fallback chain and
  ``query_lens.tolist()`` fed to decode kernels.

Every other backend reads rejection-corrected lengths directly on device and
needs no host mirror at all. The default is therefore device-native: skipping
the sync is the common path, and the legacy sync is the special case detected
through this registry.

Backend author rule: register a backend here only if its builder consumes
the exact host seq-lens mirror (e.g. passes host length lists into a kernel).
Device-native backends need no entry. A backend that subclasses a legacy
family below but builds metadata from device tensors only belongs in
``DEVICE_NATIVE_BACKEND_OVERRIDES`` (see ``fa_v1``, whose builder docstring
reads "Build from exact device lengths; CPU mirrors may be stale").
"""

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend

# Backends (with their subclasses) whose builders consume the exact host
# seq-lens mirror: the FIA/GQA base (covers C8-MXFP and 310p subclasses) and
# MLA. FlashMLA drafts route through the MLA backend and stay conservative.
LEGACY_EXACT_HOST_SEQ_LENS_BACKENDS = (AscendAttentionBackend, AscendMLABackend)

# Device-native overrides inside a legacy family.
DEVICE_NATIVE_BACKEND_OVERRIDES = (AscendFlashAttentionBackend,)


def backend_needs_exact_host_seq_lens(backend_cls: type) -> bool:
    """Return whether a backend class still consumes exact host seq-lens."""
    if issubclass(backend_cls, DEVICE_NATIVE_BACKEND_OVERRIDES):
        return False
    return issubclass(backend_cls, LEGACY_EXACT_HOST_SEQ_LENS_BACKENDS)
