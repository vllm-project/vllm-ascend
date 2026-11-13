# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""DeepSeek V4.1 cache readiness, independent of the Mooncake transport.

One instance belongs to a Push worker. Only the model thread mutates it;
sender threads receive immutable transfer snapshots from the worker instead.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

    from vllm_ascend.models.deepseek_v41.layerwise_cache_access import LayerCacheAccess

    from .metadata import PushSendReqMeta


def is_deepseek_v41(config: "VllmConfig") -> bool:
    text_config = getattr(getattr(config, "model_config", None), "hf_text_config", None)
    return getattr(text_config, "model_type", None) in ("deepseek_v41", "deepseek_v41_text")


def validate_deepseek_v41_config(config: "VllmConfig") -> None:
    parallel = config.parallel_config
    if any(
        size != 1
        for size in (
            parallel.pipeline_parallel_size,
            parallel.prefill_context_parallel_size,
            parallel.decode_context_parallel_size,
        )
    ):
        raise ValueError("Mooncake Push V4.1 requires PP=PCP=DCP=1")
    if config.speculative_config is not None:
        raise ValueError("Mooncake Push V4.1 does not support speculative decoding")


class DeepseekV41TransferPlan:
    def __init__(
        self,
        accesses: tuple["LayerCacheAccess", ...],
        block_sizes: dict[str, int],
        windows: dict[str, int],
        bounded_replay: bool,
    ) -> None:
        self.accesses = {entry.writes[0].cache_name: entry for entry in accesses}
        self.resources = {resource.cache_name for entry in accesses for resource in entry.writes}
        self.block_sizes = block_sizes
        self.windows = windows
        self.bounded_replay = bounded_replay
        self._writes: dict[str, list[torch.Tensor]] = {}
        self._coverage: dict[str, dict[str, set[int]]] = {}

    def record_write(self, cache_name: str, slots: "torch.Tensor | None", *, cache_only: bool) -> tuple[str, ...]:
        access = self.accesses[cache_name]
        if self.bounded_replay and slots is not None:
            # Slot buffers can be reused by later replay forwards. Snapshot on
            # the compute stream now, and transfer to CPU once at the step end.
            self._writes.setdefault(cache_name, []).append(slots.clone())
        if cache_only:
            return ()
        writes = access.writes[1:] if self.bounded_replay else access.writes
        return tuple(resource.cache_name for resource in writes)

    def finish_step(self, requests: dict[str, "PushSendReqMeta"]) -> dict[str, str]:
        """Validate replay's final SWA window before publishing its readiness."""
        errors = {}
        try:
            if not self.bounded_replay:
                return errors
            written = {
                name: {slot for mapping in mappings for slot in mapping.tolist() if slot >= 0}
                for name, mappings in self._writes.items()
            }
            for request_id, request in requests.items():
                end = request.num_computed_tokens + request.num_scheduled_tokens
                final = end >= request.num_prompt_tokens
                coverage = self._coverage.setdefault(request_id, {})
                for name, access in self.accesses.items():
                    size = self.block_sizes[name]
                    blocks = request.local_block_ids[access.writes[0].group_id]
                    first = max(0, request.num_prompt_tokens - self.windows[name] + 1)
                    expected = set()
                    for token in range(first, min(end, request.num_prompt_tokens)):
                        index, offset = divmod(token, size)
                        if index >= len(blocks) or blocks[index] == 0:
                            if final:
                                errors[request_id] = f"Missing V4.1 source SWA block: {name}, token={token}"
                            continue
                        expected.add(blocks[index] * size + offset)
                    covered = coverage.setdefault(name, set())
                    covered.intersection_update(expected)
                    covered.update(expected.intersection(written.get(name, ())))
                    if final and covered != expected:
                        errors[request_id] = f"Incomplete V4.1 replay SWA window: {name}"
                if final:
                    self.discard(request_id)
            return errors
        finally:
            self._writes.clear()

    def discard(self, request_id: str) -> None:
        self._coverage.pop(request_id, None)
