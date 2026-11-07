# SPDX-License-Identifier: Apache-2.0
"""Compose independent leased VMM mappings into global chunk pointer tables."""

import json
import time
from pathlib import Path

import torch

from vllm_ascend.engram_shard_layout import layout_identity, row_ranges, validate_owner_set

from .shared_pool import SharedPoolClient, _close_buffers, _private


class ShardedPool:
    """One worker's owner sessions, shared across its Engram layers."""

    def __init__(self, root, identity, *, device, library_path=None):
        self.root = Path(root)
        _private(self.root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            manifest.get("schema") != 1
            or manifest.get("identity") != identity
            or manifest.get("state") not in ("loading", "ready")
        ):
            raise ValueError("Incompatible sharded Engram manifest")
        self.membership = validate_owner_set(manifest["membership"])
        self.layout_id = layout_identity(self.membership, identity)
        self.clients, self.references = {}, 0
        self.closed = False
        self.device, self.library_path = device, library_path
        self.identity = identity
        self.pending = []
        self.deadline = manifest.get("expires_at", 0)
        self.attached = False
        if manifest.get("layout_id") != self.layout_id:
            raise ValueError("Incompatible Engram shard layout")
        if manifest["state"] == "loading":
            if not isinstance(self.deadline, (int, float)) or self.deadline <= time.time():
                raise ValueError("Expired Engram load plan")
            return
        self._attach(manifest)

    def _attach(self, manifest):
        try:
            if manifest["layout_id"] != self.layout_id or len(manifest["generations"]) != len(
                self.membership["owners"]
            ):
                raise ValueError("Incompatible Engram shard layout")
            for rank in range(len(self.membership["owners"])):
                client = SharedPoolClient(
                    self.root / f"owner-{rank}", self.identity, device=self.device, library_path=self.library_path
                )
                self.clients[rank] = client
                expected = dict(layout_id=self.layout_id, owner_rank=rank, membership=self.membership)
                if client.manifest.get("sharding") != expected:
                    raise ValueError("Engram shard owner/layout mismatch")
                if client.manifest["generation"] != manifest["generations"][rank]:
                    raise ValueError("Engram owner generation changed after bootstrap")
            self._validate_tables()
            self.attached = True
        except BaseException:
            self.close()
            raise

    def finalize(self):
        """Attach after dense/draft loading, before any profiling or graph use."""
        if self.closed:
            raise RuntimeError("Sharded Engram pool is closed")
        if self.attached:
            return
        while True:
            manifest = json.loads((self.root / "manifest.json").read_text())
            if (
                manifest.get("schema") != 1
                or manifest.get("identity") != self.identity
                or manifest.get("membership") != self.membership
                or manifest.get("layout_id") != self.layout_id
            ):
                raise ValueError("Engram load plan changed during model loading")
            if manifest.get("state") == "failed":
                raise RuntimeError("Engram owner bootstrap failed")
            if manifest.get("state") == "ready":
                break
            if manifest.get("state") != "loading" or time.time() >= self.deadline:
                raise TimeoutError("Engram owners did not finish loading")
            time.sleep(0.1)
        self._attach(manifest)
        try:
            for placeholder, key, start, end, spec in self.pending:
                if placeholder.tensor is None:
                    continue
                view = self.view(key, start, end, **spec)
                placeholder.parts, placeholder.ptrs = view.parts, view.ptrs
            self.pending.clear()
        except BaseException:
            for placeholder, *_ in self.pending:
                placeholder.close()
            self.close()
            raise

    def _validate_tables(self):
        keys = set(self.clients[0].manifest["tensors"])
        if not keys:
            raise ValueError("Empty Engram sharded table set")
        for client in self.clients.values():
            if set(client.manifest["tensors"]) != keys:
                raise ValueError("Engram owners expose different tables")
        for key in keys:
            first = self.clients[0].manifest["tensors"][key]
            spans = row_ranges(first["global_rows"], len(self.clients), self.membership["chunk_rows"])
            for rank, (start, end) in enumerate(spans):
                meta = self.clients[rank].manifest["tensors"][key]
                if (meta.get("global_rows"), meta.get("row_start"), meta["rows"], meta["width"], meta["dtype"]) != (
                    first["global_rows"],
                    start,
                    end - start,
                    first["width"],
                    first["dtype"],
                ):
                    raise ValueError("Engram shards do not cover one consistent table")

    def acquire(self):
        if self.closed:
            raise RuntimeError("Sharded Engram pool is closed")
        self.references += 1
        return _TableLease(self)

    def view(self, key, start, end, *, rows, width, dtype):
        if self.closed or not 0 <= start < end <= rows:
            raise ValueError("Invalid sharded Engram view")
        if not self.attached:
            view = _GlobalView([], None, (end - start, width), dtype, self.membership["chunk_rows"])
            self.pending.append((view, key, start, end, dict(rows=rows, width=width, dtype=dtype)))
            return view
        parts = []
        try:
            for client in self.clients.values():
                meta = client.manifest["tensors"][key]
                if meta["global_rows"] != rows:
                    raise ValueError("Engram global row count mismatch")
                parts.append(
                    client.view(
                        key,
                        0,
                        meta["rows"],
                        rows=meta["rows"],
                        width=width,
                        dtype=dtype,
                        chunk_rows=self.membership["chunk_rows"],
                    )
                )
            pointers = torch.cat([part.ptrs for part in parts])
            return _GlobalView(parts, pointers, (end - start, width), dtype, self.membership["chunk_rows"])
        except BaseException:
            for part in parts:
                part.close()
            raise

    def close(self):
        if self.closed:
            return
        if self.clients:
            _close_buffers(self.clients, self.library_path)
        self.closed = True


def finalize_shared_engram(*models):
    """Finalize each shared pool once, including speculative model modules."""
    seen = set()
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            lease = getattr(module, "_shared_pool_client", None)
            pool = getattr(lease, "pool", None)
            if isinstance(pool, ShardedPool) and id(pool) not in seen:
                seen.add(id(pool))
                pool.finalize()


def close_shared_engram(*models):
    """Stop device access before releasing shared views and owner leases."""
    tables = []
    seen = set()
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if getattr(module, "_shared_pool_client", None) is not None and id(module) not in seen:
                seen.add(id(module))
                tables.append(module)
    if not tables:
        return
    # A failed synchronize is not proof that unmapping is safe.
    torch.npu.synchronize()
    errors = []
    for table in tables:
        try:
            table.close_host_offload()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(f"Failed to close {len(errors)} shared Engram tables; leases may remain") from errors[0]


class _TableLease:
    def __init__(self, pool):
        self.pool, self.views = pool, []
        self.closed = False

    def view(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError("Engram table lease is closed")
        view = self.pool.view(*args, **kwargs)
        self.views.append(view)
        return view

    def close(self):
        if self.closed:
            return
        for view in self.views:
            view.close()
        if self.pool.references == 1:
            self.pool.close()
        self.pool.references -= 1
        self.closed = True


class _GlobalView:
    global_rows = True

    def __init__(self, parts, pointers, shape, dtype, chunk_rows):
        self.parts, self.ptrs = parts, pointers
        self.tensor = torch.empty(shape, dtype=dtype, device="meta")
        self.chunk_rows = chunk_rows

    def close(self):
        for part in self.parts:
            part.close()
        self.ptrs = self.tensor = None
