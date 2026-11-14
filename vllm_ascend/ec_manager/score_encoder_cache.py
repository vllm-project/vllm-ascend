# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


class ScoreEncoderCache:

    def __init__(
        self,
        encoder_cache: "EncoderCache",
        device: torch.device,
    ) -> None:
        self.encoder_cache = encoder_cache
        self.device = device
        self.tmp_encoder_cache: dict[str, torch.Tensor] = {}
        self.cpu_encoder_cache: dict[str, torch.Tensor] = {}
        self._pending_encoder_cache_copies: deque[
            tuple[torch.Tensor, torch.npu.Event]
        ] = deque()
        self._current_step_metadata: (
            "EncoderCacheManagerMetadata | None") = None
        self._req_refs: dict[str, set[str]] = {}
        self._req_hash_index: dict[str, set[str]] = {}

    def has(self, mm_hash: str) -> bool:
        return (
            mm_hash in self.encoder_cache.encoder_outputs
            or mm_hash in self.tmp_encoder_cache
            or mm_hash in self.cpu_encoder_cache
        )

    def get(self, mm_hash: str) -> torch.Tensor | None:
        encoder_output = self.encoder_cache.encoder_outputs.get(mm_hash, None)
        if encoder_output is not None:
            return encoder_output

        encoder_output = self.tmp_encoder_cache.get(mm_hash, None)
        if encoder_output is not None:
            return encoder_output

        cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
        if cpu_value is None:
            return None

        encoder_output = self._copy_cpu_encoder_cache_to_device(
            cpu_value, non_blocking=False
        )
        self.tmp_encoder_cache[mm_hash] = encoder_output
        return encoder_output

    def store(self, mm_hash: str, output: torch.Tensor) -> None:
        metadata = self._current_step_metadata
        promoting_mm_hashes = (
            metadata.promoting_mm_hashes if metadata is not None else []
        )
        npu_freed = metadata.npu_freed if metadata is not None else []

        staging = torch.empty_like(output, device="cpu", pin_memory=True)
        staging.copy_(output.detach(), non_blocking=True)
        self.cpu_encoder_cache[mm_hash] = staging

        if mm_hash in promoting_mm_hashes and mm_hash not in npu_freed:
            self.encoder_cache.encoder_outputs[mm_hash] = output
        else:
            self.tmp_encoder_cache[mm_hash] = output

    def apply_metadata(self, metadata: "EncoderCacheManagerMetadata") -> None:
        self._current_step_metadata = metadata
        self._clear_finished_encoder_cache_copies()

        for mm_hash in metadata.npu_freed:
            self.encoder_cache.encoder_outputs.pop(mm_hash, None)

        for mm_hash in metadata.cpu_freed:
            self.cpu_encoder_cache.pop(mm_hash, None)

        for mm_hash in metadata.promoting_mm_hashes:
            cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
            if cpu_value is None:
                continue
            self.encoder_cache.encoder_outputs[mm_hash] = (
                self._copy_cpu_encoder_cache_to_device(cpu_value)
            )

        for mm_hash in metadata.cpu_get_encoder_mm_hashes:
            if (
                mm_hash in self.encoder_cache.encoder_outputs
                or mm_hash in self.tmp_encoder_cache
            ):
                continue
            cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
            if cpu_value is None:
                continue
            self.tmp_encoder_cache[mm_hash] = (
                self._copy_cpu_encoder_cache_to_device(cpu_value)
            )

    def track_request(
        self,
        req_id: str,
        mm_features: "list[MultiModalFeatureSpec]",
    ) -> None:
        hashes = {f.identifier for f in mm_features}
        self._req_hash_index[req_id] = hashes
        for mm_hash in hashes:
            self._req_refs.setdefault(mm_hash, set()).add(req_id)

    def on_request_removed(self, req_id: str) -> None:
        for mm_hash in self._req_hash_index.pop(req_id, ()):
            refs = self._req_refs.get(mm_hash)
            if refs is None:
                continue
            refs.discard(req_id)
            if not refs:
                del self._req_refs[mm_hash]
                self.tmp_encoder_cache.pop(mm_hash, None)

    def reset(self) -> None:
        if (
            self.tmp_encoder_cache
            or self.cpu_encoder_cache
            or self._pending_encoder_cache_copies
        ):
            torch.npu.synchronize()
        self.encoder_cache.encoder_outputs.clear()
        self.tmp_encoder_cache.clear()
        self.cpu_encoder_cache.clear()
        self._req_refs.clear()
        self._req_hash_index.clear()
        self._current_step_metadata = None
        self._pending_encoder_cache_copies.clear()

    def _clear_finished_encoder_cache_copies(self) -> None:
        while (
            self._pending_encoder_cache_copies
            and self._pending_encoder_cache_copies[0][1].query()
        ):
            self._pending_encoder_cache_copies.popleft()

    def _copy_cpu_encoder_cache_to_device(
        self,
        cpu_value: torch.Tensor,
        *,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        if not non_blocking:
            return cpu_value.to(self.device, non_blocking=False)

        self._clear_finished_encoder_cache_copies()
        source = cpu_value if cpu_value.is_pinned() else cpu_value.pin_memory()
        npu_value = torch.empty_like(source, device=self.device)
        npu_value.copy_(source, non_blocking=True)

        copy_done = torch.npu.Event()
        copy_done.record(torch.npu.current_stream())
        self._pending_encoder_cache_copies.append((source, copy_done))
        npu_value.record_stream(torch.npu.current_stream())
        return npu_value
