#!/usr/bin/env python3
"""DCP-only adaptation on top of PR #16208 (e5662f7). No DIAG logs."""
import sys

FILE = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py"
with open(FILE) as f:
    c = f.read()

def patch(old, new, tag):
    global c
    if old not in c:
        print(f"MISS: {tag}"); return False
    if new in c:
        print(f"SKIP: {tag}"); return True
    c = c.replace(old, new, 1)
    print(f"OK: {tag}"); return True

ok = True

# 1. save-leader helper (insert before _process_save_for_layer_batch)
ok &= patch(
    "    def _process_save_for_layer_batch(",
    '''    def _is_layerwise_save_leader(self) -> bool:
        """Exactly one rank per (pcp, dcp, head_or_tp) group saves/allocates.

        Without DCP the plain ``tp_rank % put_step == 0`` dedup is correct
        because all TP ranks hold identical KV. With DCP the context is
        sharded: ranks sharing (dcp_rank, head_or_tp_rank) hold the same
        shard, while the put_step groups span DIFFERENT shards -- the plain
        dedup would drop every non-zero DCP shard from the pool. The leader
        is the smallest tp_rank inside the rank's own (dcp, head) group.
        """
        if self.dcp_size <= 1:
            return self.tp_rank % self.put_step == 0
        head = self.tp_rank // self.put_step
        peers = [
            t
            for t in range(head * self.put_step, (head + 1) * self.put_step)
            if t % self.dcp_size == self.dcp_rank
        ]
        if not peers:
            return True
        return self.tp_rank == min(peers)

    def _process_save_for_layer_batch(''',
    "1: save-leader helper")

# 2. apply in _process_save_for_layer_batch
ok &= patch(
    """        # Only the first rank in each put_step group saves to the
        # pool.  Other ranks in the same group share the same KV cache
        # (e.g. MLA latent), so they skip save to avoid redundant writes.
        # TODO(lf): Distribute KV block writes across ranks in the put_step group.
        if self.tp_rank % self.put_step != 0:
            return""",
    """        # Only one rank in each (pcp, dcp, head_or_tp) group saves to the
        # pool. Other ranks in the same group share the same KV shard
        # (e.g. MLA latent replicated across the non-DCP TP ranks), so they
        # skip save to avoid redundant writes. With DCP>1 the plain put_step
        # dedup would drop every non-zero DCP shard (see
        # _is_layerwise_save_leader).
        if not self._is_layerwise_save_leader():
            return""",
    "2: leader in _process_save_for_layer_batch")

# 3. apply in _alloc_gvas_for_save
ok &= patch(
    """        if not self.use_layerwise_transfer:
            return
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            return
        if self.tp_rank % self.put_step != 0:
            return""",
    """        if not self.use_layerwise_transfer:
            return
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            return
        if not self._is_layerwise_save_leader():
            return""",
    "3: leader in _alloc_gvas_for_save")

# 4. region size: DCP shard-major with aligned stride
ok &= patch(
    """        per_layer = sum(gbl) // n_local
        n_global = max(total_layers, int(self.num_layers), n_local)
        return per_layer * n_global""",
    """        per_layer = sum(gbl) // n_local
        n_global = max(total_layers, int(self.num_layers), n_local)
        # DCP shard-major layout: ranks sharing head_or_tp_rank (put_step>1,
        # e.g. MLA) hold different context shards of the same logical block
        # and share ONE region, so the region must cover all shards. The
        # per-shard stride is the group byte total aligned up to the GVA
        # hugepage size: hybrid DSA groups have non-uniform per-layer bytes,
        # and an unaligned stride would yield misaligned GVA addresses that
        # SDMA rejects. With put_step == 1 every rank owns a distinct region
        # key and no shard separation is needed.
        cp_scale = getattr(self, "pcp_size", 1) * getattr(self, "dcp_size", 1)
        if self.put_step > 1 and cp_scale > 1:
            gva_align = 2 * 1024 * 1024
            shard_stride = (sum(gbl) + gva_align - 1) // gva_align * gva_align
            return shard_stride * cp_scale
        return per_layer * n_global""",
    "4: region size x cp_scale (aligned)")

# 5. builder offset: DCP shard base (aligned) on top of PP offset
ok &= patch(
    """            # Global byte offset for the shared GVA region: each stage writes
            # its local layers at (pp_layer_offset + local_layer) * per_layer_bytes.
            layer_byte_offset = 0
            if getattr(self, "pp_size", 1) > 1:
                gbl = self.group_block_len.get(group_id) or []
                if gbl and group_num_layers > 0:
                    per_layer = sum(gbl) // group_num_layers
                    layer_byte_offset = int(getattr(self, "layerwise_key_layer_offset", 0)) * per_layer""",
    """            # Global byte offsets for the shared GVA region:
            #   PP: each stage writes its local layers at
            #       (pp_layer_offset + local_layer) * per_layer_bytes.
            #   DCP (shard-major): ranks sharing head_or_tp_rank (put_step>1)
            #       hold different context shards of the same logical block
            #       and share ONE region; shard d occupies the byte range
            #       [d * shard_stride, (d+1) * shard_stride) where the stride
            #       is the group byte total aligned up to the GVA hugepage
            #       size (unaligned strides yield misaligned GVA addresses).
            #       With put_step == 1 every rank owns a distinct region key
            #       and no shard separation is needed.
            layer_byte_offset = 0
            gbl = self.group_block_len.get(group_id) or []
            if gbl and group_num_layers > 0:
                per_layer = sum(gbl) // group_num_layers
                if getattr(self, "pp_size", 1) > 1:
                    layer_byte_offset += int(getattr(self, "layerwise_key_layer_offset", 0)) * per_layer
                cp_scale = getattr(self, "pcp_size", 1) * getattr(self, "dcp_size", 1)
                if cp_scale > 1 and self.put_step > 1:
                    gva_align = 2 * 1024 * 1024
                    shard_stride = (sum(gbl) + gva_align - 1) // gva_align * gva_align
                    shard_idx = getattr(self, "pcp_rank", 0) * self.dcp_size + getattr(self, "dcp_rank", 0)
                    layer_byte_offset += shard_idx * shard_stride""",
    "5: builder shard offset (aligned)")

with open(FILE, "w") as f:
    f.write(c)
print("ALL DCP PATCHES APPLIED" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)