# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Submit the Kimi decoder's independent recurrent-state update alongside DSpark."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState, _postprocess_recoverssm_align_kernel

from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext, MultiGroupKDACommit


class KimiDSparkRecoverSSMState(RecoverSSMState):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self._bindings: tuple[KDARecoverSSMCommitContext, ...] = ()
        self._merged: MultiGroupKDACommit | None = None
        self._active = False
        self._submitted = False
        self._stream = torch.npu.Stream(device=device)
        self._ready = torch.npu.Event()
        self._done = torch.npu.Event()

    def _can_merge_step(self) -> bool:
        step = self._step
        if not step or len(step) < 2:
            return False
        groups = []
        for metadata in step:
            if not isinstance(metadata, KimiK3KDAMetadata):
                return False
            if not metadata.num_spec_decodes or metadata.num_spec_decodes != step[0].num_spec_decodes:
                return False
            commit = metadata.recoverssm_commit
            context = metadata.recoverssm_context
            if commit is None or commit.align is None or not isinstance(context, KDARecoverSSMCommitContext):
                return False
            # Mixed batches index sampler rows through the Kimi builder's
            # explicit map; a pure batch can use an implicit identity map.
            if (metadata.num_prefills or metadata.num_decodes) and commit.request_indices is None:
                return False
            groups.append(context)

        if len(groups) != len(self._bindings) or any(a is not b for a, b in zip(groups, self._bindings)):
            bindings = tuple(groups)
            merged = MultiGroupKDACommit.create(bindings)
            self._bindings, self._merged = bindings, merged
        return self._merged is not None

    @contextmanager
    def draft_scope(self) -> Iterator[None]:
        if not self._can_merge_step():
            yield
            return
        self._active = True
        try:
            yield
        finally:
            self._active = False
            # Also fence a proposal that raised before its normal return.
            self.wait_for_commit()

    def wait_for_commit(self) -> None:
        if self._submitted:
            torch.npu.current_stream(self.device).wait_event(self._done)
            self._submitted = False

    @contextmanager
    def postprocess_stream(
        self, idx_mapping: torch.Tensor, num_sampled: torch.Tensor | int, num_computed_tokens: torch.Tensor | None
    ) -> Iterator[None]:
        if not self._active or isinstance(num_sampled, int):
            yield
            return
        # post_update has published last_sampled and computed-token counts on
        # the main stream. The draft and the following update can now fork.
        self._ready.record(torch.npu.current_stream(self.device))
        self._stream.wait_event(self._ready)
        with torch.npu.stream(self._stream):
            try:
                # Some metadata is staged on a third stream. Track its storage
                # until the last consumer even if a step exits via an exception.
                for tensor in (idx_mapping, num_sampled, num_computed_tokens):
                    if tensor is not None:
                        tensor.record_stream(self._stream)
                for metadata in self._step:
                    commit = metadata.recoverssm_commit
                    for tensor in (
                        commit.state_indices,
                        commit.query_start_loc,
                        commit.request_indices,
                        commit.align.block_table,
                        commit.align.num_computed_tokens,
                    ):
                        if tensor is not None:
                            tensor.record_stream(self._stream)
                yield
            finally:
                self._done.record(self._stream)
                self._submitted = True

    def commit_step(
        self,
        num_sampled: torch.Tensor | int,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        # A verified batch can still commit together when the scheduler skips
        # the next draft: in that case the two launches stay on the main stream.
        if isinstance(num_sampled, int) or (not self._active and not self._can_merge_step()):
            return super().commit_step(
                num_sampled, idx_mapping, state_indices=state_indices, num_accepted_tokens=num_accepted_tokens
            )

        step = self._step
        self._step = None
        assert self._merged is not None and step is not None and state_indices is not None
        self._merged.commit(step, num_sampled)
        for metadata in step:
            commit = metadata.recoverssm_commit
            align = commit.align
            _postprocess_recoverssm_align_kernel[(metadata.num_spec_decodes,)](
                idx_mapping,
                num_sampled,
                commit.request_indices,
                align.num_computed_tokens,
                state_indices,
                num_accepted_tokens,
                MAMBA_BLOCK_SIZE=align.block_size,
                BLOCK_TABLE_WIDTH=align.block_table.shape[1],
            )
