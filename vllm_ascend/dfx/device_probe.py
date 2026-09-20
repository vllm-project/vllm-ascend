# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded device evidence, outside graph capture and model execution.

Copies are queued on the current producer stream, including pinned D2H. This
intentionally trades stream bandwidth/latency for a stable snapshot. There is no
new synchronize, barrier, collective or model invocation. It is NOT zero cost.
Only owned CPU copies are inspected after an event reports completion.
"""

import dataclasses
import hashlib
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MAX_METADATA_DEPTH = 4
MAX_METADATA_FIELDS = 64
METADATA_FIELDS = frozenset(
    {
        "slot_mapping",
        "block_table",
        "block_tables",
        "seq_lens",
        "seq_lens_list",
        "query_start_loc",
        "state_indices",
        "state_indices_tensor",
        "non_spec_state_indices_tensor",
        "spec_state_indices_tensor",
        "num_accepted_tokens",
        "num_actual_tokens",
        "num_decode_tokens",
        "num_prefills",
        "num_decodes",
        "max_query_len",
        "prefill",
        "decode",
        "prefill_metadata",
        "decode_metadata",
        "non_spec_query_start_loc",
        "spec_query_start_loc",
        "non_spec_prefill_metadata",
        "non_spec_decode_metadata",
        "spec_decode_metadata",
        "causal_conv1d",
        "spec_causal_conv1d",
        "cache_indices",
    }
)


@dataclasses.dataclass
class V2BoundaryMetadata:
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_actual_tokens: int
    plain_attention: bool
    mla: bool = False


def tensor_descriptor(tensor):
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "dtype": str(tensor.dtype),
        "storage_offset": tensor.storage_offset(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "element_size": tensor.element_size(),
    }


class DeviceProbe:
    def __init__(self, recorder, runner, caches, device_api=None, cache_layout="NHD"):
        self.recorder = recorder
        self.config = recorder.config
        self.api = device_api if device_api is not None else torch.npu
        self.caches = caches
        self.cache_layout = cache_layout
        first_cache = next(iter(caches.values()), None)
        first_tensor = first_cache[0] if isinstance(first_cache, (list, tuple)) else first_cache
        self.device = first_tensor.device if first_tensor is not None else torch.device("cpu")
        self.thread_context = threading.local()
        self.last_blocks = {}
        self.groups = {
            layer: group_id
            for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            for layer in group.layer_names
        }
        self.layers = list(self.config.device_capture_layers or sorted(caches))
        unknown = set(self.layers) - caches.keys()
        if unknown:
            raise ValueError(f"DFX device_capture_layers are not allocated on this worker: {sorted(unknown)}")
        self.pending = deque()
        self.active = None
        self.quarantine = []
        self.lock = threading.Lock()
        self.sequence = 0
        self.disabled = False
        self.counters = dict(submitted=0, completed=0, busy=0, errors=0, omitted=0, skipped=0)
        self.reference = None
        if self.config.reference_trace:
            from vllm_ascend.dfx.inspect_trace import read_trace

            path = Path(self.config.reference_trace.replace("{rank}", str(recorder.identity.get("rank", 0))))
            if path.stat().st_size > self.config.max_buffer_bytes:
                raise ValueError("DFX reference trace exceeds max_buffer_bytes")
            self.reference = read_trace(path)
        layouts = {}
        for layer, cache in caches.items():
            tensors = cache if isinstance(cache, (list, tuple)) else (cache,)
            layouts[layer] = [tensor_descriptor(tensor) for tensor in tensors]
        recorder.record_event("kv_layout", {"layers": layouts, "groups": self.groups, "cache_layout": cache_layout})

    def has_pending(self):
        return bool(self.pending)

    def begin_v2(self, runner, batch, block_tables, slot_mappings, requests, *, kind, graph_mode):
        """Capture MRV2 device-assembled inputs without reading device scalars.

        Page selection is a scheduler-derived hint. Slot/table evidence comes
        from actual prepare_attn outputs, before backend-specific transformation.
        """
        if (
            self.disabled
            or not self.config.device_capture_interval
            or (self.recorder.execution_id % self.config.device_capture_interval)
        ):
            return None
        if runner.pcp_manager is not None or getattr(runner, "ubatch_runner", None) is not None:
            self.counters["skipped"] += 1
            self.recorder.record_event("device_capture_gap", {"reason": "v2_cp_or_ubatch_stream_ordering_unverified"})
            return None
        tables, metadata = [], {}
        req_ids = batch.req_ids[: batch.num_reqs] if kind == "real" else []
        for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            kernel_size = runner.block_tables.kernel_block_sizes[group_id]
            logical_size = runner.block_tables.block_sizes[group_id]
            ratio = logical_size // kernel_size
            rows, lengths = [], []
            for req in req_ids:
                state = requests.get(req)
                ids = state.block_ids[group_id] if state is not None else ()
                rows.append((ids[0] * ratio, ids[-1] * ratio + ratio - 1) if ids else (0, 0))
                lengths.append(2 if ids else 0)
            array = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
            tables.append(
                SimpleNamespace(
                    block_size=kernel_size,
                    num_blocks_per_row=lengths,
                    get_numpy_array=lambda array=array: array,
                    is_circular=False,
                    use_hybrid_blocks=ratio != 1,
                )
            )
            # Restrict mathematical slot checks to explicit ordinary FA/MLA
            # specs. Recurrent/compressed/circular state needs private metadata.
            spec = group.kv_cache_spec
            for layer in group.layer_names:
                layer_spec = getattr(spec, "kv_cache_specs", {}).get(layer, spec)
                plain = type(layer_spec).__name__ in ("FullAttentionSpec", "MLAAttentionSpec", "AttentionSpec")
                metadata[layer] = V2BoundaryMetadata(
                    slot_mapping=slot_mappings[group_id],
                    block_tables=block_tables[group_id],
                    query_start_loc=batch.query_start_loc,
                    seq_lens=batch.seq_lens,
                    num_actual_tokens=batch.num_tokens,
                    plain_attention=plain,
                    mla="MLA" in type(layer_spec).__name__,
                )
        proxy = SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=req_ids, num_reqs=len(req_ids), block_table=SimpleNamespace(block_tables=tables)
            ),
            requests=requests,
            use_dcp=runner.use_dcp,
            pcp_size=2 if runner.pcp_manager is not None else 1,
            speculative_config=runner.speculative_config,
            vllm_config=runner.vllm_config,
            # V2's KVPP/offload producer may execute on another stream.
            sparse_kv_offload_enabled=(
                getattr(runner.kvpp, "scheduler", None) is not None
                or getattr(runner, "sparse_kv_offload_enabled", False)
            ),
        )
        frame = self.begin(proxy, metadata, batch.input_ids, batch.positions, kind=kind, graph_mode=graph_mode)
        if frame is not None:
            frame["request_context_complete"] = False
        return frame

    def _copy(self, frame, name, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        size = tensor.numel() * tensor.element_size()
        if size > frame["remaining"]:
            frame["omissions"].append(name)
            self.counters["omitted"] += 1
            return
        # Keep source and destination alive through the completion event, even
        # if a later capture operation raises. Never hand live device views to IO.
        source = tensor.detach().clone(memory_format=torch.contiguous_format)
        cpu = torch.empty_like(source, device="cpu", pin_memory=source.device.type != "cpu")
        frame["copies"][name] = (source, cpu, tensor_descriptor(tensor))
        frame["remaining"] -= size
        cpu.copy_(source, non_blocking=True)

    def _metadata(self, frame, value, path="metadata", depth=0):
        if depth > MAX_METADATA_DEPTH or len(frame["copies"]) >= MAX_METADATA_FIELDS:
            frame["omissions"].append(path)
            return
        if isinstance(value, torch.Tensor):
            self._copy(frame, path, value)
        elif dataclasses.is_dataclass(value):
            # GDN attaches some backend-specific dataclasses dynamically; use
            # the explicit allowlist rather than only dataclasses.fields().
            for name in sorted(METADATA_FIELDS):
                if hasattr(value, name):
                    self._metadata(frame, getattr(value, name), f"{path}.{name}", depth + 1)
        elif type(value) in (int, bool, float, str) or value is None:
            frame["scalars"][path] = value
        elif (
            type(value) in (list, tuple)
            and len(value) <= self.config.max_checked_tokens
            and all(type(x) is int for x in value)
        ):
            frame["scalars"][path] = tuple(value)

    def begin(self, runner, metadata, input_ids, positions, *, kind="real", graph_mode="NONE", skip=False):
        if self.disabled or skip or not self.layers:
            self.counters["skipped"] += 1
            return None
        execution_id = self.recorder.execution_id
        if not self.config.device_capture_interval or execution_id % self.config.device_capture_interval:
            return None
        if self.device.type != "cpu" and self.api.is_current_stream_capturing():
            self.counters["skipped"] += 1
            return None
        executor = getattr(runner, "device_metadata_executor", None)
        if executor is not None and executor.submission_in_flight:
            # DSA metadata is produced on another stream and its original waits
            # are inside the model. Do not race that producer or add a new wait.
            self.counters["skipped"] += 1
            self.recorder.record_event("device_capture_gap", {"reason": "metadata_producer_not_joined"})
            return None
        if not self.lock.acquire(blocking=False):
            self.counters["busy"] += 1
            return None
        try:
            if self.active is not None or len(self.pending) >= self.config.device_pending_limit:
                self.counters["busy"] += 1
                return None
            self.sequence += 1
            layer = self.layers[(self.sequence - 1) % len(self.layers)]
            group_id = self.groups[layer]
            tables = runner.input_batch.block_table.block_tables
            table = tables[group_id]
            req_ids = tuple(runner.input_batch.req_ids[: runner.input_batch.num_reqs]) if kind == "real" else ()
            # Select allocated physical/kernel rows only. Prefer both old prefix
            # and current tail; do not read every KV page in a production step.
            blocks = []
            if kind == "real":
                for row in range(len(req_ids)):
                    ids = table.get_numpy_array()[row, : table.num_blocks_per_row[row]]
                    if len(ids):
                        blocks.extend((int(ids[0]), int(ids[-1])))
            else:
                # Include a previously active page as well as the reserved page.
                # This is bounded sampling, not a guarantee of detecting all idle writes.
                blocks = [0, *self.last_blocks.get(layer, ())]
            blocks = tuple(dict.fromkeys(blocks))[: self.config.device_capture_blocks]
            if kind == "real":
                self.last_blocks[layer] = blocks
            selected = metadata.get(layer) if isinstance(metadata, dict) else None
            fingerprints = []
            for req_id in req_ids:
                request = runner.requests.get(req_id)
                prompt = getattr(request, "prompt_token_ids", None)
                output = getattr(request, "output_token_ids", None)
                if prompt is None or output is None or len(prompt) + len(output) > self.config.max_checked_tokens:
                    break
                history = hashlib.sha256()
                for tokens in (prompt, output):
                    history.update(len(tokens).to_bytes(8, "little"))
                    history.update(np.asarray(tokens, dtype=np.int64).tobytes())
                lora = getattr(request, "lora_request", None)
                history.update(str(getattr(lora, "lora_int_id", None)).encode())
                fingerprints.append(history.hexdigest())
            frame = dict(
                execution_id=execution_id,
                probe_id=self.sequence,
                layer=layer,
                group=group_id,
                kind=kind,
                graph_mode=str(graph_mode),
                req_ids=req_ids,
                request_fingerprints=tuple(fingerprints),
                request_context_complete=len(fingerprints) == len(req_ids),
                blocks=blocks,
                block_size=table.block_size,
                copies={},
                scalars={},
                omissions=[],
                remaining=self.config.device_capture_bytes // 2,
                metadata_type=type(selected).__name__,
                token_axis=1
                if self.cache_layout in ("LBHNC", "HND")
                and type(selected).__name__ != "AscendMLAMetadata"
                and not (isinstance(selected, V2BoundaryMetadata) and selected.mla)
                else 0,
                recurrent=(
                    type(selected).__name__ == "GDNAttentionMetadata"
                    and not runner.use_dcp
                    and getattr(runner, "pcp_size", 1) == 1
                    and runner.speculative_config is None
                    and str(graph_mode).endswith("NONE")
                ),
                cache_safe=(
                    getattr(getattr(runner, "vllm_config", None), "kv_transfer_config", None) is None
                    and not getattr(runner, "sparse_kv_offload_enabled", False)
                ),
                simple=(
                    (
                        type(selected).__name__ in ("AscendMetadata", "AscendMLAMetadata")
                        or isinstance(selected, V2BoundaryMetadata)
                        and selected.plain_attention
                    )
                    and not runner.use_dcp
                    and getattr(runner, "pcp_size", 1) == 1
                    and runner.speculative_config is None
                    and not getattr(runner, "use_compress", False)
                    and not table.is_circular
                    and not table.use_hybrid_blocks
                    and str(graph_mode).endswith("NONE")
                ),
            )
            self.active = frame
            self._copy(frame, "input_ids", input_ids)
            self._copy(frame, "positions", positions)
            self._metadata(frame, selected)
            self._cache(frame, "before")
            return frame
        except Exception:
            self.counters["errors"] += 1
            if self.active is not None:
                self.active["omissions"].append("capture_begin_failed")
                self._seal(self.active)
            return None
        finally:
            self.lock.release()

    def _cache(self, frame, phase):
        if not frame["cache_safe"]:
            frame["omissions"].append(f"kv.{phase}:external_writer_ordering_unverified")
            return
        cache = self.caches[frame["layer"]]
        tensors = cache if isinstance(cache, (list, tuple)) else (cache,)
        for component, tensor in enumerate(tensors):
            for block in frame["blocks"]:
                name = f"kv.{phase}.{component}.{block}"
                if tensor.ndim < 1 or not 0 <= block < tensor.shape[0]:
                    frame["omissions"].append(name + ":block_out_of_range")
                    continue
                # Axis zero is the allocated block axis of MRV1's reshaped
                # per-component views, including page-strided recurrent states.
                self._copy(frame, name, tensor[block])

    def end(self, frame, logits=None):
        if frame is None:
            return
        # Runner begin/end are serialized; polling never touches active frames.
        try:
            frame["remaining"] += self.config.device_capture_bytes // 2
            if logits is not None and logits.ndim == 2 and logits.shape[0] and logits.shape[1]:
                row = (frame["probe_id"] - 1) % logits.shape[0]
                width = min(logits.shape[1], self.config.device_capture_bytes // logits.element_size())
                start = ((frame["probe_id"] - 1) * width) % logits.shape[1]
                stop = min(start + width, logits.shape[1])
                frame["scalars"]["logits_finite_selection"] = (row, start, stop)
                self._copy(frame, "raw_logits_finite", torch.isfinite(logits[row : row + 1, start:stop]).all(dim=-1))
            self._cache(frame, "after")
            if logits is not None and logits.ndim == 2 and logits.shape[0] and logits.shape[1]:
                # KV is the priority. Use leftover budget for a rotating logits
                # row/vocabulary interval; its exact coordinates are evidence.
                width = min(logits.shape[1], frame["remaining"] // logits.element_size())
                if width:
                    row = (frame["probe_id"] - 1) % logits.shape[0]
                    start = ((frame["probe_id"] - 1) * width) % logits.shape[1]
                    stop = min(start + width, logits.shape[1])
                    frame["scalars"]["logits_selection"] = (row, start, stop)
                    self._copy(frame, "raw_logits", logits[row : row + 1, start:stop])
                else:
                    frame["omissions"].append("raw_logits:budget")
        except Exception:
            self.counters["errors"] += 1
            frame["omissions"].append("capture_end_failed")
        finally:
            # One producer appends; poll is the only consumer. deque append and
            # popleft are thread-safe; never wait for the polling lock here.
            self._seal(frame)

    def _seal(self, frame):
        try:
            event = self.api.Event()
            event.record(self.api.current_stream())
            self.pending.append((event, frame))
            self.counters["submitted"] += 1
        except Exception:
            # An async copy may already be issued. Retain buffers and stop
            # allocating rather than recycling memory whose DMA status is unknown.
            self.quarantine.append(frame)
            self.disabled = True
            self.counters["errors"] += 1
        self.active = None

    def poll(self):
        if self.disabled:
            return
        if not self.lock.acquire(blocking=False):
            return
        ready = []
        try:
            if self.device.type != "cpu" and not getattr(self.thread_context, "initialized", False):
                self.api.set_device(self.device)
                self.thread_context.initialized = True
            if self.pending:
                event, frame = self.pending[0]
                if not event.query():
                    return
                self.pending.popleft()
                ready.append(frame)
        except Exception:
            self.counters["errors"] += 1
            self.disabled = True
        finally:
            self.lock.release()
        for frame in ready:
            try:
                payload, findings = self._inspect(frame)
                self.recorder.record_event("device_snapshot", payload, tuple(findings), frame["execution_id"])
                self.counters["completed"] += 1
            except Exception:
                self.counters["errors"] += 1

    def _inspect(self, frame):
        cpu = {name: item[1] for name, item in frame["copies"].items()}
        findings = []
        tensors = {}
        for name, (_, value, descriptor) in frame["copies"].items():
            raw = value.reshape(-1).view(torch.uint8).numpy().copy()
            tensors[name] = {"layout": descriptor, "data": raw, "data_layout": "contiguous_logical_values"}
            if name == "raw_logits" and not bool(torch.isfinite(value).all()):
                findings.append("raw_logits_nonfinite")
            if name == "raw_logits_finite" and not bool(value.all()):
                findings.append("raw_logits_nonfinite")
            # Whole blocks include invalid tail bytes; report nonfinite content
            # as evidence only, not a definite valid-token numerical failure.
            if name.startswith("kv.after") and value.is_floating_point():
                tensors[name]["nonfinite_elements_including_unused"] = int((~torch.isfinite(value)).sum())
        slots = cpu.get("metadata.slot_mapping")
        if any(
            value.dtype not in (torch.int32, torch.int64)
            for name, value in cpu.items()
            if name in ("metadata.slot_mapping", "metadata.block_tables", "metadata.query_start_loc")
        ):
            findings.append("device_metadata_dtype")
        if slots is not None:
            slots = slots.to(torch.int64).numpy().reshape(-1)
            if frame["kind"] == "sync_only" and (slots >= 0).any():
                findings.append("sync_only_writable_slots")
            if frame["simple"]:
                n = frame["scalars"].get("metadata.num_actual_tokens", len(slots))
                if n < 0 or n > len(slots):
                    findings.append("device_slot_count_mismatch")
                    n = len(slots)
                valid = slots[:n][slots[:n] >= 0]
                cache = self.caches[frame["layer"]]
                component = cache[0] if isinstance(cache, (tuple, list)) else cache
                capacity = component.shape[0] * frame["block_size"]
                if (slots < -1).any() or (valid >= capacity).any():
                    findings.append("device_slot_out_of_range")
                if (valid < frame["block_size"]).any():
                    findings.append("device_write_to_null_block")
                if len(np.unique(valid)) != len(valid):
                    findings.append("device_slot_write_collision")
                if (slots[n:] != -1).any():
                    findings.append("device_padding_writable_slots")
                positions = cpu.get("positions")
                starts = cpu.get("metadata.query_start_loc")
                block_table = cpu.get("metadata.block_tables")
                if positions is not None and starts is not None and block_table is not None and positions.ndim == 1:
                    pos = positions.to(torch.int64).numpy()
                    offsets = starts.to(torch.int64).numpy()
                    table = block_table.to(torch.int64).numpy()
                    rows = len(frame["req_ids"])
                    if offsets.ndim != 1 or table.ndim != 2:
                        findings.append("device_metadata_shape")
                    elif len(offsets) < rows + 1 or offsets[0] != 0 or (np.diff(offsets[: rows + 1]) < 0).any():
                        findings.append("device_query_offsets_invalid")
                    elif offsets[rows] != n or len(pos) < n or len(table) < rows:
                        findings.append("device_query_count_mismatch")
                    else:
                        for row in range(rows):
                            start, stop = offsets[row : row + 2]
                            indices = pos[start:stop] // frame["block_size"]
                            if (indices < 0).any() or (indices >= table.shape[1]).any():
                                findings.append("device_position_block_out_of_range")
                                continue
                            expected = table[row, indices] * frame["block_size"] + pos[start:stop] % frame["block_size"]
                            if not np.array_equal(expected, slots[start:stop]):
                                findings.append("device_position_slot_mismatch")
        for name, before in cpu.items():
            if not name.startswith("kv.before."):
                continue
            after = cpu.get(name.replace(".before.", ".after."))
            if after is None:
                continue
            changed = (
                (before.reshape(-1).view(torch.uint8) != after.reshape(-1).view(torch.uint8))
                .reshape(*before.shape, before.element_size())
                .any(dim=-1)
            )
            if frame["kind"] == "sync_only" and bool(changed.any()):
                findings.append("sync_only_kv_changed")
            elif frame["recurrent"]:
                component, block = map(int, name.split(".")[-2:])
                if component == 0:
                    prefixes = (
                        "metadata.non_spec_prefill_metadata.causal_conv1d",
                        "metadata.non_spec_decode_metadata.causal_conv1d",
                    )
                    pairs = [
                        (cpu.get(prefix + ".cache_indices"), cpu.get(prefix + ".query_start_loc"))
                        for prefix in prefixes
                    ]
                elif component == 1:
                    pairs = [
                        (
                            cpu.get("metadata.non_spec_state_indices_tensor"),
                            cpu.get("metadata.non_spec_query_start_loc"),
                        )
                    ]
                else:
                    pairs = []
                active = set()
                verified = False
                for indices, offsets in pairs:
                    if (
                        indices is not None
                        and offsets is not None
                        and indices.ndim == offsets.ndim == 1
                        and len(offsets) == len(indices) + 1
                    ):
                        verified = True
                        lengths = offsets[1:] - offsets[:-1]
                        if bool((lengths < 0).any()):
                            findings.append("recurrent_query_offsets_invalid")
                        active.update(indices[(lengths > 0) & (indices >= 0)].tolist())
                if 0 in active:
                    findings.append("recurrent_write_to_null_block")
                if verified and block in active:
                    tensors[name.replace(".before.", ".after.")]["whole_state_valid"] = True
                    if after.is_floating_point() and not bool(torch.isfinite(after).all()):
                        findings.append("recurrent_state_nonfinite")
                elif verified and bool(changed.any()):
                    findings.append("recurrent_state_changed_without_active_query")
            elif frame["simple"] and slots is not None and before.ndim == 3:
                # Use the runner's explicit cache-layout contract, never infer
                # the token axis from a shape (head count can equal block size).
                axis = frame["token_axis"]
                if before.shape[axis] != frame["block_size"]:
                    continue
                block = int(name.rsplit(".", 1)[1])
                allowed = np.zeros(frame["block_size"], dtype=bool)
                local = slots[(slots >= block * frame["block_size"]) & (slots < (block + 1) * frame["block_size"])]
                allowed[local % frame["block_size"]] = True
                valid = allowed.copy()
                seq_lens = cpu.get("metadata.seq_lens")
                lengths = (
                    seq_lens.reshape(-1).tolist()
                    if seq_lens is not None
                    else frame["scalars"].get("metadata.seq_lens_list")
                )
                table = cpu.get("metadata.block_tables")
                validity_source = "write_slots_only"
                if lengths is not None and table is not None and table.ndim == 2:
                    for row, length in enumerate(lengths[: min(len(frame["req_ids"]), table.shape[0])]):
                        if length < 0 or length > table.shape[1] * frame["block_size"]:
                            findings.append("device_sequence_length_out_of_range")
                            continue
                        used = min((length + frame["block_size"] - 1) // frame["block_size"], table.shape[1])
                        for index in torch.nonzero(table[row, :used] == block).reshape(-1).tolist():
                            valid[: min(frame["block_size"], length - index * frame["block_size"])] = True
                    validity_source = "seq_lens_and_block_table"
                captured = tensors[name.replace(".before.", ".after.")]
                captured["checked_token_offsets"] = np.flatnonzero(valid).tolist()
                captured["token_axis"] = axis
                captured["validity_source"] = validity_source
                unexpected = changed.movedim(axis, 0)[torch.from_numpy(~allowed)]
                if bool(unexpected.any()):
                    findings.append("kv_changed_outside_planned_slots")
                if after.is_floating_point() and bool(
                    (~torch.isfinite(after.movedim(axis, 0)[torch.from_numpy(allowed)])).any()
                ):
                    findings.append("written_kv_nonfinite")
                if after.is_floating_point() and bool(
                    (~torch.isfinite(after.movedim(axis, 0)[torch.from_numpy(valid)])).any()
                ):
                    findings.append("valid_kv_nonfinite")
        digest = hashlib.sha256()
        for name in ("input_ids", "positions"):
            if name in tensors:
                digest.update(str((tensors[name]["layout"]["shape"], tensors[name]["layout"]["dtype"])).encode())
                digest.update(tensors[name]["data"].tobytes())
        payload = {
            key: frame[key]
            for key in (
                "probe_id",
                "layer",
                "group",
                "kind",
                "graph_mode",
                "req_ids",
                "request_fingerprints",
                "request_context_complete",
                "blocks",
                "block_size",
                "metadata_type",
                "simple",
                "recurrent",
                "cache_safe",
                "scalars",
                "omissions",
            )
        }
        payload.update(
            tensors=tensors,
            input_digest=digest.hexdigest(),
            input_digest_complete="input_ids" in tensors and "positions" in tensors,
            scope="runner_boundary_not_kernel_internal",
            snapshot_phase="before_and_after_forward",
            exact_replay=False,
        )
        if self.reference is not None:
            from vllm_ascend.dfx.inspect_trace import compare_traces

            payload["reference"] = compare_traces(
                {
                    "manifest": self.recorder._manifest,
                    "identity": self.recorder.identity,
                    "records": [
                        {"kind": "device_snapshot", "payload": payload, "execution_id": frame["execution_id"]},
                    ],
                },
                self.reference,
                atol=self.config.reference_atol,
                rtol=self.config.reference_rtol,
            )
            if payload["reference"]["status"] == "mismatch":
                findings.append("reference_numeric_mismatch")
        return payload, sorted(set(findings))
