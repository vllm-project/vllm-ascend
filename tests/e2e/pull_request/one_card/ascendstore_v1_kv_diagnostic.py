# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMPORARY byte-exact Bulk KV audit. Remove after archiving native CI evidence.

This deliberately synchronizes and copies device memory, and optionally poisons
only the allocated Load destination. It is not a performance or race-freedom test.
"""

import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from hashlib import sha256
from typing import Any

import numpy as np
import torch

from tests.e2e.pull_request.one_card.ascendstore_v1_diagnostic import (
    AscendStoreDiagnosticWorkerExtension,
    first_divergence,
    request_index,
    snapshot_batch,
    token_batch_context,
)

AUDIT_BLOCK_SIZE = 128
AUDIT_PREFIX_TOKENS = 8192
AUDIT_REQUESTS = 8


class AscendStoreKVByteDiagnosticWorkerExtension(AscendStoreDiagnosticWorkerExtension):
    def start_ascendstore_kv_byte_audit(self) -> dict[str, Any]:
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        assert not hasattr(self, "_kv_byte_audit"), "Previous KV audit is still installed"
        connector = get_kv_transfer_group()
        runner = self.model_runner
        config = runner.vllm_config
        assert config.parallel_config.tensor_parallel_size == config.parallel_config.pipeline_parallel_size == 1
        assert config.scheduler_config.async_scheduling is False
        assert config.model_config.enforce_eager
        assert config.speculative_config is None
        kind = type(connector).__name__
        if kind == "AscendStoreConnector":
            worker = connector.connector_worker
            assert not worker.use_layerwise and not worker.load_async
            caches = worker.kv_caches
            implementation = "production"
        else:
            assert kind == "AscendStoreV1Connector", kind
            worker = connector.worker
            assert not connector._use_layerwise
            assert type(worker._load_timeline).__name__ == "SynchronousLoadTimeline"
            caches = worker._resources.kv_caches
            implementation = "v1"
        assert len(caches) == config.model_config.hf_text_config.num_hidden_layers
        self._kv_byte_audit = KVByteAudit(
            runner,
            connector,
            caches,
            implementation,
            torch.npu.synchronize,
            AUDIT_BLOCK_SIZE,
            AUDIT_PREFIX_TOKENS,
            AUDIT_REQUESTS,
        )
        self._kv_byte_audit.layout.update(
            {
                "device": torch.npu.get_device_name(),
                "torch_version": str(torch.__version__),
                "model_config_sha256": sha256(config.model_config.hf_text_config.to_json_string().encode()).hexdigest(),
                "model_revision": config.model_config.revision,
            }
        )
        self._kv_byte_audit.install()
        return self._kv_byte_audit.layout

    def begin_ascendstore_kv_byte_phase(self, phase_id: str, mode: str) -> dict[str, str]:
        self._kv_byte_audit.begin_phase(phase_id, mode)
        return {"phase_id": phase_id, "mode": mode}

    def end_ascendstore_kv_byte_phase(self) -> dict[str, Any]:
        return self._kv_byte_audit.end_phase()

    def stop_ascendstore_kv_byte_audit(self) -> dict[str, bool]:
        audit = self._kv_byte_audit
        try:
            audit.close()
        finally:
            del self._kv_byte_audit
        return {"hooks_restored": True, "temporary_reference_closed": True}


class KVByteAudit:
    """Compare actual consumer block-table locations against a disk-backed cold reference.

    The reference is captured before the first Store submission. A Load is checked
    both before model execution and after its first forward; the last prompt token
    is compared separately because it is recomputed, not transferred.
    """

    def __init__(
        self,
        runner: Any,
        connector: Any,
        caches: dict[str, Any],
        implementation: str,
        synchronize: Callable[[], None],
        block_size: int,
        prefix_tokens: int,
        expected_requests: int,
    ) -> None:
        assert sys.byteorder == "little", "BF16 diagnostics assume a little-endian host"
        assert prefix_tokens > 0 and prefix_tokens % block_size == 0
        assert runner.block_tables.block_sizes == runner.block_tables.kernel_block_sizes == [block_size]
        self.runner = runner
        self.connector = connector
        self.implementation = implementation
        self.synchronize = synchronize
        self.block_size = block_size
        self.prefix_tokens = prefix_tokens
        self.prefix_blocks = prefix_tokens // block_size
        self.expected_requests = expected_requests
        self.entries: list[tuple[str, torch.Tensor]] = []
        for layer_name, pair in sorted(caches.items()):
            assert isinstance(pair, (tuple, list)) and len(pair) == 2, "Only ordinary K/V attention is in scope"
            for role, cache in zip(("K", "V"), pair, strict=True):
                assert cache.dtype == torch.bfloat16 and cache.ndim == 4 and cache.shape[1] == block_size
                assert cache[0].is_contiguous() and cache.stride(0) >= cache[0].numel(), (
                    "Expected dense NHD block payloads"
                )
                self.entries.append((f"{layer_name}/{role}", cache))
        assert self.entries
        self.prefix_bytes_per_request = sum(cache[0].numel() * cache.element_size() for _, cache in self.entries)
        self.prefix_bytes_per_request *= self.prefix_blocks
        reference_bytes = self.prefix_bytes_per_request * expected_requests
        assert shutil.disk_usage(tempfile.gettempdir()).free > reference_bytes * 2, "Insufficient reference disk space"
        self.reference_owner = ExitStack()
        # This context spans multiple Worker RPCs; close() releases it after the paired Load phases.
        self.reference = self.reference_owner.enter_context(tempfile.TemporaryFile())  # noqa: SIM115
        self.references: dict[int, dict[str, Any]] = {}
        self.phase: dict[str, Any] | None = None
        self.batch: dict[str, Any] | None = None
        self.pending_forward: dict[str, tuple[int, list[int]]] = {}
        self.original_prepare = runner._prepare_inputs_impl
        self.original_load = connector.start_load_kv
        self.original_save = connector.wait_for_save
        self.layout = {
            "implementation": implementation,
            "block_size": block_size,
            "prefix_tokens": prefix_tokens,
            "expected_requests": expected_requests,
            "prefix_bytes_per_request": self.prefix_bytes_per_request,
            "comparison": "full byte equality; SHA-256 is recorded as supporting evidence, not the comparison oracle",
            "entries": [
                {
                    "name": name,
                    "dtype": str(cache.dtype),
                    "shape": list(cache.shape),
                    "stride": list(cache.stride()),
                    "storage_offset_bytes": cache.storage_offset() * cache.element_size(),
                    "block_payload_bytes": cache[0].numel() * cache.element_size(),
                    "physical_block_stride_bytes": cache.stride(0) * cache.element_size(),
                }
                for name, cache in self.entries
            ],
        }

    def install(self) -> None:
        self.runner._prepare_inputs_impl = self.prepare_inputs
        self.connector.start_load_kv = self.start_load
        self.connector.wait_for_save = self.before_store

    def begin_phase(self, phase_id: str, mode: str) -> None:
        assert self.phase is None and not self.pending_forward
        assert mode in ("record", "observe", "poison")
        assert (mode == "record") == (not self.references), "Record once, then compare the same reference"
        self.batch = None
        self.phase = {"phase_id": phase_id, "mode": mode, "requests": []}

    def prepare_inputs(self, *args: Any, **kwargs: Any) -> Any:
        batch = self.original_prepare(*args, **kwargs)
        if self.phase is not None:
            marker = f"ascendstore-diag-{self.phase['phase_id']}-p"
            assert all(marker in request_id for request_id in batch.req_ids), "Unmeasured request entered the audit"
            self.batch = snapshot_batch(batch, 0)
        return batch

    def start_load(self, *args: Any, **kwargs: Any) -> Any:
        rows = self.prompt_rows() if self.phase is not None and self.phase["mode"] != "record" else []
        for request_id, index, blocks in rows:
            self.validate_plan(request_id, blocks, load=True)
            assert index in self.references, "A loaded request has no independent cold reference"
            if self.phase is not None and self.phase["mode"] == "poison":
                self.poison_destination(blocks)
        result = self.original_load(*args, **kwargs)
        if rows:
            self.synchronize()
        for request_id, index, blocks in rows:
            observation = self.compare_request(index, blocks, after_forward=False)
            assert self.phase is not None
            observation["request_id"] = request_id
            observation["poison_verified"] = self.phase["mode"] == "poison"
            self.phase["requests"].append(observation)
            self.pending_forward[request_id] = (index, blocks)
            self.require_equal_prefix(observation)
        return result

    def before_store(self, *args: Any, **kwargs: Any) -> Any:
        if self.phase is None:
            return self.original_save(*args, **kwargs)
        rows = self.prompt_rows()
        if rows:
            self.synchronize()
        for request_id, index, blocks in rows:
            if self.phase["mode"] == "record":
                self.validate_plan(request_id, blocks, load=False)
                assert index not in self.references, "The cold reference must never be overwritten"
                observation = self.capture_request(request_id, index, blocks)
            else:
                assert self.pending_forward.pop(request_id) == (index, blocks)
                observation = self.compare_request(index, blocks, after_forward=True)
                self.require_equal_prefix(observation)
            self.phase["requests"].append(observation)
        return self.original_save(*args, **kwargs)

    def prompt_rows(self) -> list[tuple[str, int, list[int]]]:
        assert self.batch is not None and self.phase is not None
        rows = []
        for row, request_id in enumerate(self.batch["request_ids"]):
            computed = self.batch["computed_tokens"][row]
            scheduled = self.batch["scheduled_tokens"][row]
            assert self.batch["prompt_lengths"][row] == self.prefix_tokens + 1
            if computed + scheduled != self.prefix_tokens + 1:
                continue
            assert computed == (0 if self.phase["mode"] == "record" else self.prefix_tokens)
            # Read the actual model's gathered table, independently of the connector's transfer ranges.
            blocks = self.runner.block_tables.input_block_tables[0][row, : self.prefix_blocks + 1].cpu().tolist()
            assert len(blocks) == self.prefix_blocks + 1 and len(set(blocks)) == len(blocks)
            assert all(0 < block < self.entries[0][1].shape[0] for block in blocks)
            rows.append((request_id, request_index(request_id), blocks))
        return rows

    def validate_plan(self, request_id: str, blocks: list[int], *, load: bool) -> None:
        metadata = self.connector._get_connector_metadata()
        if self.implementation == "v1":
            commands = metadata.load.commands if load else metadata.store.source_pending_commands
            matches = [command for command in commands if command.request_id == request_id]
            assert len(matches) == 1, "Expected one complete synchronous transfer command"
            command = matches[0]
            span = command.load_range if load else command.store_range
            assert span.start_token == 0 and span.end_token >= self.prefix_tokens
            if load:
                assert span.end_token == self.prefix_tokens
            else:
                assert command.num_prompt_tokens == self.prefix_tokens + 1
                assert not any(c.request_id == request_id for c in metadata.store.source_ready_commands)
            groups = command.block_ids_by_group
        else:
            matches = [request for request in metadata.requests if request.req_id == request_id]
            assert len(matches) == 1
            request = matches[0]
            if load:
                assert request.load_spec is not None and request.load_spec.can_load
                assert request.load_spec.vllm_cached_tokens == 0
                assert request.load_spec.kvpool_cached_tokens == self.prefix_tokens
            else:
                assert request.can_save and request.save_start_token == 0
                assert request.save_end_token >= self.prefix_tokens
                assert request.num_prompt_tokens == self.prefix_tokens + 1
            groups = request.block_ids_by_group
        assert len(groups) == 1 and list(groups[0][: len(blocks)]) == blocks, "Transfer/model block tables disagree"

    def capture_request(self, request_id: str, index: int, blocks: list[int]) -> dict[str, Any]:
        reference: dict[str, Any] = {"request_index": index, "request_id": request_id, "blocks": blocks, "entries": []}
        for name, cache in self.entries:
            prefix = read_prefix_bytes(cache, blocks[:-1])
            tail = read_tail_bytes(cache, blocks[-1])
            assert bf16_all_finite(prefix) and bf16_all_finite(tail), "Cold reference contains nonfinite KV"
            offset = self.reference.tell()
            assert self.reference.write(prefix + tail) == len(prefix) + len(tail)
            reference["entries"].append(
                {
                    "name": name,
                    "offset": offset,
                    "prefix_bytes": len(prefix),
                    "tail_bytes": len(tail),
                    "prefix_sha256": sha256(prefix).hexdigest(),
                    "tail_sha256": sha256(tail).hexdigest(),
                }
            )
        self.references[index] = reference
        self.reference.flush()
        return {"stage": "before_store", **reference}

    def poison_destination(self, blocks: list[int]) -> None:
        for _, cache in self.entries:
            for start, count in consecutive_runs(blocks[:-1]):
                cache.narrow(0, start, count).fill_(float("nan"))
        self.synchronize()
        for _, cache in self.entries:
            raw = read_prefix_bytes(cache, blocks[:-1])
            bits = np.frombuffer(raw, dtype="<u2")
            assert np.all((bits & 0x7F80) == 0x7F80) and np.all((bits & 0x007F) != 0), "Poison did not take effect"

    def compare_request(self, index: int, blocks: list[int], *, after_forward: bool) -> dict[str, Any]:
        source = self.references[index]
        comparisons = []
        for (name, cache), entry in zip(self.entries, source["entries"], strict=True):
            assert name == entry["name"]
            self.reference.seek(entry["offset"])
            expected = self.reference.read(entry["prefix_bytes"])
            assert len(expected) == entry["prefix_bytes"]
            comparison: dict[str, Any] = {
                "name": name,
                "prefix": compare_bytes(expected, read_prefix_bytes(cache, blocks[:-1])),
            }
            if after_forward:
                expected_tail = self.reference.read(entry["tail_bytes"])
                actual_tail = read_tail_bytes(cache, blocks[-1])
                comparison["recomputed_last_prompt_token"] = compare_bytes(expected_tail, actual_tail)
                comparison["recomputed_last_prompt_token"].update(bf16_difference(expected_tail, actual_tail))
            comparisons.append(comparison)
        return {
            "stage": "after_forward" if after_forward else "after_load_before_forward",
            "request_index": index,
            "source_blocks": source["blocks"],
            "destination_blocks": blocks,
            "prefix_bytes": sum(entry["prefix"]["bytes"] for entry in comparisons),
            "different_prefix_bytes": sum(entry["prefix"]["different_bytes"] for entry in comparisons),
            "entries": comparisons,
        }

    @staticmethod
    def require_equal_prefix(observation: dict[str, Any]) -> None:
        if observation["different_prefix_bytes"]:
            print("ASCENDSTORE_KV_BYTE_MISMATCH " + json.dumps(observation), flush=True)
            raise AssertionError("Restored/consumed KV differs byte-for-byte from the pre-Store reference")

    def end_phase(self) -> dict[str, Any]:
        assert self.phase is not None and not self.pending_forward, "The model did not consume every audited Load"
        result = self.phase
        self.phase = None
        stages = ("before_store",) if result["mode"] == "record" else ("after_load_before_forward", "after_forward")
        for stage in stages:
            indices = [row["request_index"] for row in result["requests"] if row["stage"] == stage]
            assert sorted(indices) == list(range(self.expected_requests)), f"Incomplete {stage} coverage: {indices}"
        return result

    def close(self) -> None:
        self.runner._prepare_inputs_impl = self.original_prepare
        self.connector.start_load_kv = self.original_load
        self.connector.wait_for_save = self.original_save
        self.reference_owner.close()


def consecutive_runs(blocks: list[int]) -> list[tuple[int, int]]:
    assert blocks
    runs = [(blocks[0], 1)]
    for block in blocks[1:]:
        start, count = runs[-1]
        if block == start + count:
            runs[-1] = (start, count + 1)
        else:
            runs.append((block, 1))
    return runs


def read_prefix_bytes(cache: torch.Tensor, blocks: list[int]) -> bytes:
    # Preserve BF16 bits, but exclude the sibling K/V region and page padding from block-strided views.
    return b"".join(
        cache.narrow(0, start, count).detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        for start, count in consecutive_runs(blocks)
    )


def read_tail_bytes(cache: torch.Tensor, block: int) -> bytes:
    return cache[block, :1].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def compare_bytes(expected: bytes, actual: bytes) -> dict[str, Any]:
    assert len(expected) == len(actual), "A truncated byte snapshot cannot pass"
    differing = np.flatnonzero(np.frombuffer(expected, dtype=np.uint8) != np.frombuffer(actual, dtype=np.uint8))
    return {
        "bytes": len(expected),
        "equal": expected == actual,
        "different_bytes": int(differing.size),
        "first_different_byte": int(differing[0]) if differing.size else None,
        "expected_sha256": sha256(expected).hexdigest(),
        "actual_sha256": sha256(actual).hexdigest(),
    }


def bf16_all_finite(raw: bytes) -> bool:
    return bool(np.all((np.frombuffer(raw, dtype="<u2") & 0x7F80) != 0x7F80))


def bf16_difference(expected: bytes, actual: bytes) -> dict[str, Any]:
    left = (np.frombuffer(expected, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    right = (np.frombuffer(actual, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    finite = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
    return {
        "finite": finite,
        "different_elements": int(np.count_nonzero(left != right)),
        "max_absolute_difference": float(np.max(np.abs(left - right))) if finite else None,
    }


def compare_byte_audit_phases(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(p["implementation"], p["case"], p["source"], p["byte_audit_mode"]): p for p in phases}
    comparisons = []
    for right in phases:
        implementation, case, source, mode = (
            right[key] for key in ("implementation", "case", "source", "byte_audit_mode")
        )
        references = []
        if implementation == "v1":
            references.append(("production-v1", ("production", case, source, mode)))
        if mode in ("record", "observe"):
            references.append(("byte-audit-effect", (implementation, case, source, "off")))
        if mode == "poison":
            references.append(("poison-effect", (implementation, case, source, "observe")))
        if source == "external":
            references.append(("cold-external", (implementation, case, "cold", "off" if mode == "off" else "record")))
        for contrast, key in references:
            left = by_key[key]
            differences = []
            for a, b in zip(left["observations"], right["observations"], strict=True):
                assert a["request_index"] == b["request_index"]
                difference = first_divergence(a, b)
                if difference is not None:
                    index = a["request_index"]
                    for side, phase in (("left", left), ("right", right)):
                        difference[side]["batch"] = token_batch_context(
                            phase["batches"], index, difference["token_position"]
                        )
                    differences.append({"request_index": index, **difference})
            comparisons.append(
                {"contrast": contrast, "left": left["id"], "right": right["id"], "differences": differences}
            )
    return comparisons
