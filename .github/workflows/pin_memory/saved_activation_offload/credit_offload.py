# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Credit-bounded saved-tensor activation offload for the benchmark harness."""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch
from torch.autograd.graph import saved_tensors_hooks


class CreditTimeout(RuntimeError):
    """A credit did not become available before the configured deadline."""


class CreditDeadlock(RuntimeError):
    """A pool has no free credit and no event which can release one."""


@dataclass
class SlotMeta:
    shape: tuple
    stride: tuple
    dtype: torch.dtype
    device: torch.device
    storage_id: int
    nbytes: int
    pack_index: int


@dataclass
class Credit:
    tag: str
    layer_idx: int
    id: int
    nbytes: int = 0
    gpu_refs: list = field(default_factory=list)
    host_bufs: list = field(default_factory=list)
    slots: list = field(default_factory=list)
    d2h_start: object | None = None
    d2h_event: object | None = None
    d2h_spans: list = field(default_factory=list)
    h2d_start: object | None = None
    h2d_event: object | None = None
    consume_event: object | None = None
    release_event: object | None = None
    seen_storages: set = field(default_factory=set)
    storage_to_slot: dict = field(default_factory=dict)
    restored: list = field(default_factory=list)
    unpack_count: list = field(default_factory=list)
    gpu_bufs: list = field(default_factory=list)
    bwd_credit: object | None = None
    h2d_started: bool = False
    host_released: bool = False


@dataclass(frozen=True)
class _Ref:
    group_id: int
    slot: int


class CreditPool:
    """Host-polled credits whose resources remain held through an event."""

    def __init__(self, capacity: int, timeout_s: float = 60.0, poll_s: float = 0.0001, spin_s: float = 0.00005):
        if capacity < 1:
            raise ValueError(f"credit capacity must be >= 1, got {capacity}")
        if timeout_s <= 0:
            raise ValueError(f"credit timeout must be > 0, got {timeout_s}")
        self.capacity = capacity
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.spin_s = spin_s
        self._free = capacity
        self._next_credit_id = 0
        self._acquired = {}
        self._outstanding = deque()
        self.wait_s = 0.0
        self.max_held = 0
        self.max_held_bytes = 0
        self.max_outstanding = 0
        self.acquisitions = 0
        self.blocking_acquisitions = 0

    @property
    def held(self):
        return self.capacity - self._free

    def _release(self, credit):
        credit.gpu_refs.clear()
        self._acquired.pop(credit.id, None)
        self._free += 1

    def _reap(self):
        # Events are ordered by acquisition. Waiting for the oldest avoids
        # allowing newer layers to bypass forward backpressure.
        while self._outstanding and self._outstanding[0].release_event.query():
            self._release(self._outstanding.popleft())

    def _host_wait(self, event, deadline, tag):
        # An already-complete event is not a stall, so it must not be counted
        # as one: the blocking count is how the experiment shows that credits
        # bind in the pinned arm and not in the pageable one.
        if event.query():
            return
        start = time.perf_counter()
        self.blocking_acquisitions += 1
        spin_until = start + self.spin_s
        while not event.query():
            now = time.perf_counter()
            if now >= deadline:
                self.wait_s += now - start
                raise CreditTimeout(f"{tag}: no credit after {self.timeout_s}s")
            # Sleeping immediately would charge the sleep granularity to the
            # arm where credits actually bind, so spin over short waits first.
            if now >= spin_until:
                time.sleep(self.poll_s)
        self.wait_s += time.perf_counter() - start

    def _take_free(self, tag, layer_idx):
        self._free -= 1
        self.acquisitions += 1
        self._next_credit_id += 1
        credit = Credit(tag=tag, layer_idx=layer_idx, id=self._next_credit_id)
        self._acquired[credit.id] = credit
        self.max_held = max(self.max_held, self.held)
        return credit

    def acquire(self, tag: str, layer_idx: int):
        self._reap()
        deadline = time.perf_counter() + self.timeout_s
        while self._free == 0:
            if not self._outstanding:
                raise CreditDeadlock(f"{tag}: no free credit and none outstanding")
            self._host_wait(self._outstanding[0].release_event, deadline, tag)
            self._reap()
        return self._take_free(tag, layer_idx)

    def try_acquire(self, tag: str, layer_idx: int):
        """Acquire without blocking, for callers that must not stall compute."""
        self._reap()
        if self._free == 0:
            return None
        return self._take_free(tag, layer_idx)

    def note_held_bytes(self):
        held_bytes = sum(item.nbytes for item in self._acquired.values())
        self.max_held_bytes = max(self.max_held_bytes, held_bytes)

    def arm(self, credit, event):
        if credit.id not in self._acquired:
            raise RuntimeError(f"{credit.tag}: cannot arm an inactive credit")
        if credit.release_event is not None:
            raise RuntimeError(f"{credit.tag}: credit is already armed")
        credit.release_event = event
        self._outstanding.append(credit)
        self.max_outstanding = max(self.max_outstanding, len(self._outstanding))
        self.note_held_bytes()

    def cancel(self, credit):
        if credit.id not in self._acquired:
            return
        if credit.release_event is not None:
            raise RuntimeError(f"{credit.tag}: cannot cancel an armed credit")
        self._release(credit)

    def drain(self):
        error = None
        while self._outstanding:
            credit = self._outstanding.popleft()
            try:
                credit.release_event.synchronize()
            except Exception as exc:  # preserve cleanup for every other credit
                error = error or exc
            self._release(credit)
        for credit in list(self._acquired.values()):
            self._release(credit)
        if self._free != self.capacity:
            raise RuntimeError(f"credit pool drain left {self.held} credits held")
        if error is not None:
            raise error

    def reset_metrics(self):
        if self.held:
            raise RuntimeError("cannot reset metrics while credits are held")
        self.wait_s = 0.0
        self.max_held = 0
        self.max_held_bytes = 0
        self.max_outstanding = 0
        self.acquisitions = 0
        self.blocking_acquisitions = 0


class HostBufferPool:
    """Dense CPU buffers, recycled only after their H2D reader completes."""

    def __init__(self, accelerator, pinned):
        self.accelerator = accelerator
        self.pinned = pinned
        self._available = {}
        self._pending = []
        self._checked_out = {}
        self.allocated_bytes = 0
        self.pinned_bytes = 0
        # None, not True, in the pageable arm: there is nothing to verify, and
        # a True would read as "we checked that these buffers are pinned".
        self.is_pinned_verified = False if pinned else None

    @staticmethod
    def _key(tensor, pinned):
        return (tuple(tensor.shape), tensor.dtype, tensor.layout, pinned)

    def _reap(self, force=False):
        pending = self._pending
        self._pending = []
        for index, (key, tensor, event) in enumerate(pending):
            try:
                if force:
                    event.synchronize()
                if force or event.query():
                    self._available.setdefault(key, []).append(tensor)
                else:
                    self._pending.append((key, tensor, event))
            except Exception:
                self._pending.append((key, tensor, event))
                self._pending.extend(pending[index + 1 :])
                raise

    def get_dense(self, tensor):
        self._reap()
        key = self._key(tensor, self.pinned)
        available = self._available.get(key)
        if available:
            host = available.pop()
        else:
            host = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, layout=tensor.layout, device="cpu")
            if self.pinned:
                host = self.accelerator.pin_memory(host, make_copy=False)
                # Track it before verifying so close() still unpins the buffer
                # if the verification below raises.
                self._checked_out[id(host)] = host
                if not self.accelerator.is_pinned(host):
                    raise RuntimeError("native pin_memory returned an unpinned host buffer")
                self.is_pinned_verified = True
                self.pinned_bytes += host.numel() * host.element_size()
            self.allocated_bytes += host.numel() * host.element_size()
        self._checked_out[id(host)] = host
        return host

    def release(self, tensor, event):
        self._checked_out.pop(id(tensor), None)
        self._pending.append((self._key(tensor, self.pinned), tensor, event))

    def release_ready(self, tensor):
        self._checked_out.pop(id(tensor), None)
        key = self._key(tensor, self.pinned)
        self._available.setdefault(key, []).append(tensor)

    def drain(self):
        self._reap(force=True)

    def close(self):
        error = None
        try:
            self.drain()
        except Exception as exc:
            error = exc
        tensors = [tensor for available in self._available.values() for tensor in available]
        tensors.extend(tensor for _, tensor, _ in self._pending)
        tensors.extend(self._checked_out.values())
        if self.pinned:
            seen = set()
            for tensor in tensors:
                tensor_id = id(tensor)
                if tensor_id in seen:
                    continue
                seen.add(tensor_id)
                try:
                    self.accelerator.unpin_memory(tensor)
                except Exception as exc:
                    error = error or exc
        self._available.clear()
        self._pending.clear()
        self._checked_out.clear()
        if error is not None:
            raise error


class SavedIntermediateOffload:
    """Scope selected module saves into credit-bounded D2H/H2D groups."""

    def __init__(
        self,
        model,
        scope_modules,
        selector,
        accelerator,
        use_pin_memory,
        forward_credits=2,
        backward_credits=2,
        min_offload_bytes=1024 * 1024,
        credit_timeout_s=60.0,
    ):
        self.model = model
        self.scope_modules = dict(scope_modules)
        self.selector = set(selector)
        self.offload_layers = {layer_idx for layer_idx, _, _ in self.selector}
        self.accelerator = accelerator
        self.min_offload_bytes = min_offload_bytes
        self.fwd_pool = CreditPool(forward_credits, credit_timeout_s)
        self.bwd_pool = CreditPool(backward_credits, credit_timeout_s)
        self.host_pool = HostBufferPool(accelerator, use_pin_memory)
        self.s_copy = accelerator.Stream()
        if self.s_copy is None:
            raise RuntimeError("saved activation offload requires CUDA streams")
        self._tls = threading.local()
        self._hooks = []
        self._saved_context = None
        self._active = False
        self._groups_by_id = {}
        self._groups_by_layer = {}
        self._prefetch_order = []
        self._prefetch_pos = 0
        self._highest_layer = max(self.scope_modules)
        self._backward_started = False
        self._top_hook_seen = False
        self._backward_generation = 0
        self._scope_entry_events = []
        self._mlp_spans = []
        self.scope_forward_calls = 0
        self.offloaded_tensors = 0
        self.offloaded_bytes = 0
        self.restored_tensors = 0
        self.restored_bytes = 0
        self.host_enqueue_s = 0.0
        self._install_hooks()
        self._parameter_storages = {
            storage_identity(param) for param in model.parameters() if param.numel() and param.device.type != "meta"
        }

    def _install_hooks(self):
        for layer_idx, module in self.scope_modules.items():
            module._ds_layer_idx = layer_idx
            self._hooks.append(module.register_forward_pre_hook(self._scope_pre_hook))
            self._hooks.append(register_forward_hook_always(module, self._scope_post_hook))
            self._hooks.append(module.register_full_backward_pre_hook(self._backward_pre_hook))
            self._hooks.append(module.register_full_backward_hook(self._backward_post_hook))

    def _scope_pre_hook(self, module, args):
        if not self._active:
            return
        if getattr(self._tls, "in_scope", False):
            raise RuntimeError("nested target-module execution is not supported")
        layer_idx = module._ds_layer_idx
        self._scope_entry_events.append((layer_idx, self._record_timing_event()))
        if layer_idx not in self.offload_layers:
            # Taking a credit for a layer that never offloads would stall this
            # layer's compute only in the arm where credits bind, which would
            # bias the pinned-versus-pageable comparison.
            return
        credit = self.fwd_pool.acquire(f"fwd:L{layer_idx}", layer_idx)
        self._tls.in_scope = True
        self._tls.credit = credit
        self._tls.pack_index = 0
        self._tls.scope_input_storages = storages_of(args)
        self._tls.mlp_start = self._record_timing_event()
        self.scope_forward_calls += 1

    def _scope_post_hook(self, module, args, output):
        if not self._active or not getattr(self._tls, "in_scope", False):
            return
        credit = self._tls.credit
        self._mlp_spans.append((self._tls.mlp_start, self._record_timing_event()))
        try:
            if not credit.slots:
                self.fwd_pool.cancel(credit)
                return
            credit.d2h_event = self._record_copy_event()
            credit.restored = [False] * len(credit.slots)
            credit.unpack_count = [0] * len(credit.slots)
            self.fwd_pool.arm(credit, credit.d2h_event)
            self._groups_by_id[credit.id] = credit
            self._groups_by_layer[credit.layer_idx] = credit
        except Exception:
            if credit.host_bufs:
                # Copies may already be queued even if event creation or arm
                # failed. Synchronize the copy stream before returning those
                # buffers to the owned pool.
                self.s_copy.synchronize()
                for host in credit.host_bufs:
                    self.host_pool.release_ready(host)
                credit.host_bufs = []
            if credit.release_event is None:
                self.fwd_pool.cancel(credit)
            raise
        finally:
            self._tls.in_scope = False
            self._tls.credit = None
            self._tls.pack_index = 0
            self._tls.scope_input_storages = set()

    def _pack_hook(self, tensor):
        if not getattr(self._tls, "in_scope", False):
            return tensor
        credit = self._tls.credit
        pack_index = self._tls.pack_index
        self._tls.pack_index += 1
        key = (credit.layer_idx, pack_index, tensor_signature(tensor))
        if key not in self.selector or self._should_skip(tensor):
            return tensor
        storage_id = storage_identity(tensor)
        if storage_id in self._tls.scope_input_storages or storage_id in self._parameter_storages:
            return tensor
        previous_slot = credit.storage_to_slot.get(storage_id)
        if previous_slot is not None:
            return _Ref(credit.id, previous_slot)

        # Capture compute before entering the copy-stream context. Calling
        # current_stream() inside that context would return s_copy itself.
        compute = self.accelerator.current_stream()
        host = self.host_pool.get_dense(tensor)
        try:
            self.s_copy.wait_stream(compute)
            enqueue_start = time.perf_counter()
            with self.accelerator.stream(self.s_copy):
                if credit.d2h_start is None:
                    credit.d2h_start = self._record_timing_event()
                # Bracket each copy on its own. The group-wide span also covers
                # the compute this hook waits on between saves, so it overstates
                # transfer time and understates bandwidth.
                copy_start = self._record_timing_event()
                host.copy_(tensor.detach(), non_blocking=True)
                copy_end = self._record_timing_event()
        except Exception:
            self.s_copy.synchronize()
            self.host_pool.release_ready(host)
            raise
        credit.d2h_spans.append((copy_start, copy_end))
        self.host_enqueue_s += time.perf_counter() - enqueue_start
        tensor.record_stream(self.s_copy)

        slot = len(credit.slots)
        nbytes = tensor.numel() * tensor.element_size()
        credit.gpu_refs.append(tensor)
        credit.host_bufs.append(host)
        credit.slots.append(
            SlotMeta(
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                device=tensor.device,
                storage_id=storage_id,
                nbytes=nbytes,
                pack_index=pack_index,
            )
        )
        credit.nbytes += nbytes
        credit.seen_storages.add(storage_id)
        credit.storage_to_slot[storage_id] = slot
        self.fwd_pool.note_held_bytes()
        self.offloaded_tensors += 1
        self.offloaded_bytes += nbytes
        return _Ref(credit.id, slot)

    def _should_skip(self, tensor):
        # This predicate is adapted from DeepSpeed's activation offload engine.
        # Offset and overlapping views cannot be recreated safely from a dense
        # host buffer with only shape and stride metadata.
        nbytes = tensor.numel() * tensor.element_size()
        overlap = has_internal_overlap(tensor)
        return (
            self.accelerator.is_synchronized_device()
            or not self.accelerator.on_accelerator(tensor)
            or nbytes < self.min_offload_bytes
            or isinstance(tensor, torch.nn.Parameter)
            or (hasattr(torch.nn, "Buffer") and isinstance(tensor, torch.nn.Buffer))
            or tensor.storage_offset() != 0
            or overlap
        )

    def _seed_backward(self):
        if self._backward_started:
            return
        self._backward_started = True
        self._backward_generation = 1
        self._prefetch_order = sorted(self._groups_by_layer, reverse=True)
        self._prefetch_pos = 0
        self._fill_prefetch_window()

    def _begin_backward(self, layer_idx):
        if layer_idx != self._highest_layer:
            return
        if self._top_hook_seen:
            self._backward_generation += 1
            raise RuntimeError("a second backward on one offload generation is not supported")
        self._top_hook_seen = True
        self._seed_backward()

    def _backward_pre_hook(self, module, grad_output):
        if not self._active:
            return
        layer_idx = module._ds_layer_idx
        self._begin_backward(layer_idx)
        group = self._groups_by_layer.get(layer_idx)
        if group is not None and not group.h2d_started:
            self._issue_h2d(group)

    def _fill_prefetch_window(self):
        while self.bwd_pool.held < self.bwd_pool.capacity:
            if not self._issue_next_prefetch(blocking=False):
                return

    def _issue_next_prefetch(self, blocking=True):
        while self._prefetch_pos < len(self._prefetch_order):
            group = self._groups_by_layer[self._prefetch_order[self._prefetch_pos]]
            if group.h2d_started:
                self._prefetch_pos += 1
                continue
            if not self._issue_h2d(group, blocking=blocking):
                return False
            self._prefetch_pos += 1
            return True
        return False

    def _issue_h2d(self, group, blocking=True):
        if group.h2d_started:
            return False
        tag = f"bwd:L{group.layer_idx}"
        if blocking:
            credit = self.bwd_pool.acquire(tag, group.layer_idx)
        else:
            credit = self.bwd_pool.try_acquire(tag, group.layer_idx)
            if credit is None:
                return False
        credit.nbytes = group.nbytes
        try:
            with self.accelerator.stream(self.s_copy):
                # Order H2D after the matching D2H, but deliberately not after
                # the compute stream: waiting on queued backward compute would
                # collapse the prefetch window into just-in-time reloads.
                # Allocating inside this context tags the buffers to s_copy, so
                # the unpack hook's record_stream governs their reuse.
                self.s_copy.wait_event(group.d2h_event)
                group.gpu_bufs = [
                    torch.empty_strided(slot.shape, slot.stride, dtype=slot.dtype, device=slot.device)
                    for slot in group.slots
                ]
                group.h2d_start = self._record_timing_event()
                for dest, source in zip(group.gpu_bufs, group.host_bufs):
                    dest.copy_(source, non_blocking=True)
                group.h2d_event = self._record_timing_event()
            group.h2d_started = True
            credit.gpu_refs = group.gpu_bufs
            group.bwd_credit = credit
            return True
        except Exception:
            credit.gpu_refs.clear()
            self.bwd_pool.cancel(credit)
            group.gpu_bufs.clear()
            raise

    def _unpack_hook(self, packed):
        if not isinstance(packed, _Ref):
            return packed
        group = self._groups_by_id.get(packed.group_id)
        if group is None:
            raise RuntimeError(f"offload group {packed.group_id} is no longer active")
        if not self._backward_started:
            # PyTorch does not guarantee the top module's backward pre-hook runs
            # before an earlier unpack, so seed the prefetch window from here
            # rather than treating the later top hook as a second backward.
            self._seed_backward()
        if not group.h2d_started:
            self._issue_h2d(group)
        if not group.restored[packed.slot]:
            compute = self.accelerator.current_stream()
            compute.wait_event(group.h2d_event)
            restored = group.gpu_bufs[packed.slot]
            restored.record_stream(compute)
            group.restored[packed.slot] = True
            self.restored_tensors += 1
            self.restored_bytes += group.slots[packed.slot].nbytes
        group.unpack_count[packed.slot] += 1
        return group.gpu_bufs[packed.slot]

    def _backward_post_hook(self, module, grad_input, grad_output):
        if not self._active:
            return
        group = self._groups_by_layer.get(module._ds_layer_idx)
        if group is None or group.bwd_credit is None:
            return
        try:
            compute = self.accelerator.current_stream()
            group.consume_event = compute.record_event()
            self.bwd_pool.arm(group.bwd_credit, group.consume_event)
            for host in group.host_bufs:
                self.host_pool.release(host, group.h2d_event)
            group.host_released = True
            group.host_bufs = []
            group.bwd_credit = None
            # Replace this layer's slot without blocking. Waiting for a credit
            # here would stall the autograd thread on compute that cannot run
            # until this hook returns, draining backward at every layer.
            self._issue_next_prefetch(blocking=False)
        except Exception:
            if group.bwd_credit is not None and group.bwd_credit.release_event is None:
                self.bwd_pool.cancel(group.bwd_credit)
                group.bwd_credit = None
            raise

    def __enter__(self):
        if self._active:
            raise RuntimeError("offload context is not reentrant")
        self._active = True
        self._saved_context = saved_tensors_hooks(self._pack_hook, self._unpack_hook)
        self._saved_context.__enter__()
        return self

    def _record_timing_event(self):
        event = self.accelerator.Event(enable_timing=True)
        event.record()
        return event

    def _record_copy_event(self):
        with self.accelerator.stream(self.s_copy):
            return self._record_timing_event()

    @staticmethod
    def _elapsed_ms(start, end):
        end.synchronize()
        return start.elapsed_time(end)

    def assert_backward_consumed(self):
        """Fail loudly if a group was never reloaded and released in backward.

        A module backward hook that silently does not fire would leave the
        backward pool unbounded for the step while every timing still looks
        plausible, so the invariant is checked rather than assumed.
        """
        missing = sorted(
            group.layer_idx
            for group in self._groups_by_id.values()
            if not group.h2d_started or group.consume_event is None
        )
        if missing:
            raise RuntimeError(f"offload groups were not reloaded and consumed during backward: {missing}")

    def timing_metrics(self):
        d2h_copy_ms = 0.0
        d2h_group_span_ms = 0.0
        h2d_ms = 0.0
        for group in self._groups_by_id.values():
            for start, end in group.d2h_spans:
                d2h_copy_ms += self._elapsed_ms(start, end)
            if group.d2h_start is not None and group.d2h_event is not None:
                d2h_group_span_ms += self._elapsed_ms(group.d2h_start, group.d2h_event)
            if group.h2d_start is not None and group.h2d_event is not None:
                h2d_ms += self._elapsed_ms(group.h2d_start, group.h2d_event)
        entry_intervals_ms = []
        for (_, start), (_, end) in zip(self._scope_entry_events, self._scope_entry_events[1:]):
            entry_intervals_ms.append(self._elapsed_ms(start, end))
        mlp_spans_ms = [self._elapsed_ms(start, end) for start, end in self._mlp_spans]
        return {
            "d2h_copy_ms": d2h_copy_ms,
            "d2h_group_span_ms": d2h_group_span_ms,
            "h2d_ms": h2d_ms,
            "entry_intervals_ms": entry_intervals_ms,
            "mlp_spans_ms": mlp_spans_ms,
        }

    def _release_group_hosts(self):
        for group in self._groups_by_id.values():
            if group.host_released:
                continue
            release_event = group.h2d_event if group.h2d_event is not None else group.d2h_event
            if release_event is None:
                self.s_copy.synchronize()
                for host in group.host_bufs:
                    self.host_pool.release_ready(host)
            else:
                for host in group.host_bufs:
                    self.host_pool.release(host, release_event)
            group.host_released = True
            group.host_bufs = []

    def _drain_step(self):
        error = None
        for action in (self.fwd_pool.drain, self.bwd_pool.drain, self._release_group_hosts, self.host_pool.drain):
            try:
                action()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def __exit__(self, exc_type, exc_value, traceback):
        saved_error = None
        try:
            self._saved_context.__exit__(exc_type, exc_value, traceback)
        except Exception as exc:
            saved_error = exc
        finally:
            self._active = False
            try:
                self._drain_step()
            except Exception as exc:
                saved_error = saved_error or exc
        if saved_error is not None and exc_type is None:
            raise saved_error
        return False

    def reset_step(self):
        if self._active or self.fwd_pool.held or self.bwd_pool.held:
            raise RuntimeError("reset_step requires an inactive, drained engine")
        self._groups_by_id.clear()
        self._groups_by_layer.clear()
        self._prefetch_order = []
        self._prefetch_pos = 0
        self._backward_started = False
        self._top_hook_seen = False
        self._backward_generation = 0
        self._scope_entry_events = []
        self._mlp_spans = []
        self.scope_forward_calls = 0
        self.offloaded_tensors = 0
        self.offloaded_bytes = 0
        self.restored_tensors = 0
        self.restored_bytes = 0
        self.host_enqueue_s = 0.0
        self.fwd_pool.reset_metrics()
        self.bwd_pool.reset_metrics()

    def close(self):
        if self._active:
            raise RuntimeError("cannot close an active offload context")
        error = None
        try:
            self._drain_step()
        except Exception as exc:
            error = exc
        try:
            self.host_pool.close()
        except Exception as exc:
            error = error or exc
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
        if error is not None:
            raise error


def has_internal_overlap(tensor):
    """Report whether two indices of the tensor read the same memory.

    Such a tensor cannot be rebuilt from a dense host copy. Torch's own check
    is private, so fall back to the zero-stride test the DeepSpeed activation
    offload engine uses when it is unavailable.
    """
    if tensor.numel() <= 1:
        return False
    probe = getattr(torch, "_debug_has_internal_overlap", None)
    if probe is not None:
        return probe(tensor) != 0
    return 0 in tensor.stride()


def register_forward_hook_always(module, hook):
    """Register a forward hook that also runs when the forward raises.

    Torch before 2.2 has no always_call, where a failing forward leaves the
    scope open until the context drain reclaims it.
    """
    try:
        return module.register_forward_hook(hook, always_call=True)
    except TypeError:
        return module.register_forward_hook(hook)


def storage_identity(tensor):
    storage = tensor.untyped_storage()
    return int(getattr(storage, "_cdata", id(storage)))


def storages_of(value):
    result = set()
    if torch.is_tensor(value):
        if value.numel() and value.device.type != "meta":
            result.add(storage_identity(value))
    elif isinstance(value, dict):
        for item in value.values():
            result.update(storages_of(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            result.update(storages_of(item))
    return result


def tensor_signature(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        int(tensor.storage_offset()),
    )


@contextlib.contextmanager
def noop_saved_tensors():
    """Matching hook context for no-offload numerical comparisons."""
    with saved_tensors_hooks(lambda tensor: tensor, lambda tensor: tensor):
        yield
