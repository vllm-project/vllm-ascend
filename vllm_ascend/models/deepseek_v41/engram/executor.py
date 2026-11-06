# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward-local Engram precomputation on one capturable NPU side stream."""

import torch


class EngramMultiStreamExecutor:
    """Run layer precomputations serially; join main only at each gate.

    All device work and record/wait edges live inside model forward. FULL
    replay therefore needs neither a Python per-batch state machine nor an
    external producer. Each forward forks from the input stream and rejoins
    at every consumer; graph-pool tensors and stream tracking own the outputs.
    """

    def __init__(self, layer_ids: tuple[int, ...], *, hash_fn, engram_layers) -> None:
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("Engram executor requires unique layer IDs")
        if hash_fn is None:
            raise ValueError("Engram hash module was not attached")
        if set(engram_layers) != set(layer_ids):
            raise ValueError("Engram layers were not attached")
        self.layer_ids = tuple(sorted(layer_ids))
        self._stream = torch.npu.Stream()
        self.next_layer_ids = dict(zip(self.layer_ids, self.layer_ids[1:]))
        self._inputs_ready = torch.npu.Event()
        self._layer_ready = {layer_id: torch.npu.Event() for layer_id in layer_ids}
        self._closed = False
        self._failed = False
        self.hash_fn = hash_fn
        self.engram_layers = engram_layers
        # Only eager execution / capture uses these handles. Every forward
        # consumes all of them; FULL replay executes captured device ops with
        # this dictionary empty. Nothing is prepared by an external pre-hook.
        self._pending_fetches = {}

    def _check_open(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("Engram executor is closed or failed")

    def begin(self, *inputs, single_token_per_request=False):
        """Hash once and start the first fetch; later layers follow in FIFO order."""
        self._check_open()
        if self._pending_fetches:
            raise RuntimeError("The previous Engram forward has unfinished fetches")
        main = torch.npu.current_stream()
        self._inputs_ready.record(main)
        for tensor in inputs:
            tensor.record_stream(self._stream)
        try:
            with torch.npu.stream(self._stream):
                self._stream.wait_event(self._inputs_ready)
                hashes, mask = self.hash_fn.forward_packed(
                    *inputs,
                    single_token_per_request=single_token_per_request,
                )
                first = self.engram_layers[self.layer_ids[0]]
                self._pending_fetches[first.layer_id] = first.start_prefetch(hashes[:, first.layer_hash_index])
            return hashes, mask
        except BaseException:
            self._failed = True
            raise

    def project(self, engram, hashes):
        """Finish this layer and record readiness before launching the next fetch."""
        self._check_open()
        if engram.layer_id not in self._layer_ready:
            raise ValueError(f"Unknown Engram layer {engram.layer_id}")
        if self.engram_layers[engram.layer_id] is not engram:
            raise ValueError("Engram consumer does not own the prefetched table")
        if engram.layer_id not in self._pending_fetches:
            raise RuntimeError(f"No pending Engram fetch for layer {engram.layer_id}")
        try:
            with torch.npu.stream(self._stream):
                # Keep launch, fetch wait, dequant and Wkv on one stream. The
                # CANN callback owns temporary fetch tensors until this stage.
                handle = self._pending_fetches.pop(engram.layer_id)
                embedding = engram.finish_prefetch(handle)
                result = engram.precompute_kv(embedding)
                self._layer_ready[engram.layer_id].record(self._stream)
                # A later fetch must not become a predecessor of this ready
                # event: main waits for this layer, never for the stream tail.
                next_id = self.next_layer_ids.get(engram.layer_id)
                if next_id is not None:
                    next_engram = self.engram_layers[next_id]
                    self._pending_fetches[next_id] = next_engram.start_prefetch(hashes[:, next_engram.layer_hash_index])
            return result
        except BaseException:
            self._failed = True
            raise

    def wait(self, layer_id: int):
        self._check_open()
        if layer_id not in self._layer_ready:
            raise ValueError(f"Unknown Engram layer {layer_id}")
        main = torch.npu.current_stream()
        main.wait_event(self._layer_ready[layer_id])
        return main

    def consume(self, engram, hidden, key, value, weight, mask):
        try:
            main = self.wait(engram.layer_id)
            # Main only consumes projected K/V in the original gate order.
            # Protect cross-stream temporaries until main has finished using them.
            for tensor in (key, value, weight, mask):
                tensor.record_stream(main)
            return engram.apply_precomputed(hidden, key, value, weight, mask)
        except BaseException:
            self._failed = True
            raise

    def release(self) -> None:
        # The device joins are in forward and replay with the graph. Do not
        # add host synchronization or Python capture/replay bookkeeping here.
        self._check_open()
        if self._pending_fetches:
            raise RuntimeError("Engram forward returned with unfinished fetches")

    def _drain(self) -> None:
        first_error = None
        # An early-layer failure can leave later tables in flight. Consume their
        # callbacks before draining/destroying resources, without running Wkv.
        for layer_id in tuple(self._pending_fetches):
            try:
                with torch.npu.stream(self._stream):
                    handle = self._pending_fetches.pop(layer_id)
                    self.engram_layers[layer_id].finish_prefetch(handle)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        for stream in (self._stream, torch.npu.current_stream()):
            try:
                stream.synchronize()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise RuntimeError("Failed to drain Engram streams") from first_error

    def abort(self) -> None:
        self._failed = True
        self._drain()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._drain()
