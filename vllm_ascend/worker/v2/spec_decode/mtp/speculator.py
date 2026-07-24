# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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
# AscendMTPSpeculator is a sibling of AscendEagleSpeculator: both inherit the
# shared flat NPU draft loop from AscendAutoRegressiveSpeculator and layer it on
# their own upstream base (MTPSpeculator vs EagleSpeculator). MTP's draft uses
# MLA attention, so it overrides only the MLA-specific piece -- forwarding
# rotary positions into build_attn_metadata -- and leaves the flat loop
# inherited unchanged. Eager only: no graph-mode adaptations.
from contextlib import contextmanager
from typing import Any

import torch
import vllm.v1.worker.gpu.spec_decode.speculator as _upstream_speculator
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


@contextmanager
def build_wrapper(positions, pad):
    """Temporarily wrap build_attn_metadata to forward MLA rotary positions.

    The flat draft-metadata path (upstream ``_build_draft_attn_metadata``) calls
    ``build_attn_metadata`` without ``positions``, but MLA reads positions
    *inside* ``build_decode_metadata`` for rotary cos/sin. This caches the
    current ``build_attn_metadata``, replaces it for the duration of the block
    with one that forwards ``positions[:pad]``, and restores it on exit -- so
    MTP can reuse the flat ``super()._build_draft_attn_metadata()`` path instead
    of duplicating the ``build_attn_metadata`` call and all its arguments.

    Must run inside ``build_attn_metadata_wrapper()``, which has already
    replaced the module-level ``build_attn_metadata`` with the Ascend builder.
    """
    raw = _upstream_speculator.build_attn_metadata  # cache

    def build_attn_metadata(*args, **kwargs):
        kwargs["positions"] = positions[:pad]
        return raw(*args, **kwargs)

    try:
        _upstream_speculator.build_attn_metadata = build_attn_metadata
        yield
    finally:
        _upstream_speculator.build_attn_metadata = raw  # restore


class AscendMTPSpeculator(AscendAutoRegressiveSpeculator, MTPSpeculator):
    """Ascend MTP speculator (MLA draft, eager only).

    Inherits the shared flat NPU draft loop from AscendAutoRegressiveSpeculator
    (layered on upstream MTPSpeculator, which supplies ``load_draft_model``) and
    overrides only the MLA-specific piece: rotary ``positions`` must reach
    ``build_attn_metadata``, which the flat draft-metadata path does not
    forward. ``draft_uses_mla`` (read from ``model_config.is_deepseek_mla`` in
    ``__init__``) gates the wrap so flat (non-MLA) MTP models keep the inherited
    flat path. No graph-mode adaptations: eager rebuilds draft attention
    metadata every step, so the MLA ``.decode`` sub-object is fresh each step
    and needs no per-step mirroring.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.draft_uses_mla = self.vllm_config.model_config.is_deepseek_mla

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        num_query_per_req: int = 1,
        causal: bool = True,
    ) -> dict[str, Any] | None:
        # Flat MTP reuses the flat super() build (positions are not needed).
        if not self.draft_uses_mla:
            return super()._build_draft_attn_metadata(
                num_reqs, num_reqs_padded, num_tokens_padded, num_query_per_req, causal
            )
        # MLA: rotary positions must reach build_attn_metadata, but the flat
        # super() path does not forward them. Wrap build_attn_metadata to inject
        # positions[:num_tokens_padded] and reuse super() (no arg duplication).
        with build_wrapper(self.input_buffers.positions, num_tokens_padded):
            return super()._build_draft_attn_metadata(
                num_reqs, num_reqs_padded, num_tokens_padded, num_query_per_req, causal
            )
