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
# AscendMTPSpeculator: MLA `.decode` hooks for MTP (flat MTP auto-adapts).
from copy import copy

from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


class AscendMTPSpeculator(AscendAutoRegressiveSpeculator, MTPSpeculator):
    """Ascend MTP speculator (MLA `.decode` hooks; flat auto-adapts)."""

    # pass
    def _get_decode_base_metadata(self, attn_metadata, num_reqs_padded):
        # Rebuild fresh if live metadata is a stale prefill residual (.decode=None).
        need_fresh = any(getattr(m, "decode", "MISSING") is None for m in attn_metadata.values())
        if need_fresh:
            assert self.input_batch is not None
            return self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_reqs_padded,
            )
        return attn_metadata

    def _prepare_step_decode_fields(self, metadata, num_reqs_padded: int) -> None:
        # Per-step .decode copy + persistent block_table snapshot (async update
        # races with next add_requests mutating input_block_tables in place).
        decode_meta = getattr(metadata, "decode", None)
        if decode_meta is not None:
            metadata.decode = copy(decode_meta)
            input_bts = self.block_tables.input_block_tables
            assert len(input_bts) == 1, "MTP/MLA draft expects a single KV cache group"
            src = input_bts[0]
            buf = getattr(self, "_draft_block_table_buf", None)
            if buf is None or buf.shape[0] < src.shape[0]:
                buf = src.new_zeros(src.shape)
                self._draft_block_table_buf = buf
            buf[:num_reqs_padded].copy_(src[:num_reqs_padded])
            metadata.decode.block_table = buf[:num_reqs_padded]

    def _write_decode_step_fields(self, metadata, query_lens_list, seq_lens_list) -> None:
        decode_meta = getattr(metadata, "decode", None)
        if decode_meta is not None:
            decode_meta.actual_seq_lengths_q = query_lens_list
            decode_meta.seq_lens_list = seq_lens_list
