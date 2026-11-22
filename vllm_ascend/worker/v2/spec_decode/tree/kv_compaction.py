import torch
from vllm.logger import init_logger

from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.kv_binding import bind_linear_kv
from vllm_ascend.worker.v2.spec_decode.tree.transport import TreeRowMove, relocate_tree_rows

logger = init_logger("vllm." + __name__)


class TreeKVCacheCompactor:
    """Move accepted-path target KV from tree slots onto the linear prefix.

    Verification writes KV at packed ``prefix+node``. Accepted drafts
    start at ``prefix+1``; the root remains at the prefix.

    Single-stream: ``run`` after target+reject and before
    ``num_computed`` increments. One ACLGraph per ``num_reqs`` so the
    per-layer gather/scatter is one replay instead of many launches.
    ``idx_mapping`` / ``path_node_ids`` are copied into persistent
    buffers; block tables and ``num_computed`` are live tensors captured
    by address. Graph gather uses preallocated scratch. Missing gears or
    capture failure fall back to the same eager ops.
    """

    def __init__(self, runner, *, bind_groups=bind_linear_kv):
        self.runner = runner
        self.device = runner.device
        self.spec_len = runner.num_speculative_steps
        self.max_num_reqs = runner.max_num_reqs
        self._graphs: dict[int, object] = {}
        self._groups = None
        self.bind_groups = bind_groups
        self._idx = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        self._path = torch.full(
            (self.max_num_reqs, self.spec_len),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._active = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=self.device)
        self._depth = torch.arange(self.spec_len, device=self.device, dtype=torch.int32) + 1

    def capture(self, sizes: list[int]) -> None:
        if not sizes or self.runner.model_config.enforce_eager:
            return
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            return
        from vllm.platforms import current_platform

        self._groups = self.bind_groups(self)
        self._idx.fill_(0)
        self._path.fill_(-1)
        self._active.zero_()
        pool = current_platform.get_global_graph_pool()
        for num_reqs in sizes:
            try:
                with torch.inference_mode():
                    self._compact(num_reqs)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, pool=pool):
                        self._compact(num_reqs)
                self._graphs[num_reqs] = graph
                logger.info("Captured tree KV compact ACLGraph num_reqs=%s", num_reqs)
            except Exception as exc:
                logger.warning(
                    "Tree KV compact ACLGraph capture failed for num_reqs=%s; eager fallback for this size. %s",
                    num_reqs,
                    exc,
                )

    def run(self, idx_mapping, acceptance: TreeAcceptance) -> None:
        path_node_ids = acceptance.path_node_ids
        num_reqs = path_node_ids.shape[0]
        if num_reqs > self.max_num_reqs or path_node_ids.shape[1] != self.spec_len:
            raise RuntimeError("Accepted path exceeds KV compaction capacity")
        self._idx[:num_reqs].copy_(idx_mapping[:num_reqs])
        self._path[:num_reqs].copy_(path_node_ids[:num_reqs])
        self._active[:num_reqs].copy_(acceptance.active)
        if num_reqs == 0:
            return
        self._groups = self.bind_groups(self, num_reqs)
        graph = self._graphs.get(num_reqs)
        if graph is not None:
            graph.replay()
        else:
            self._compact(num_reqs)

    def _compact(self, num_reqs: int) -> None:
        if not self._groups:
            return
        idx = self._idx[:num_reqs]
        path = self._path[:num_reqs]
        num_computed = self.runner.req_states.num_computed_tokens.gpu
        nslot = num_reqs * self.spec_len
        live_req = (idx >= 0) & (idx < num_computed.shape[0])
        safe_idx = idx.clamp(0, num_computed.shape[0] - 1).long()
        prefix = num_computed[safe_idx]
        dst_pos = prefix.unsqueeze(1) + self._depth
        src_pos = prefix.unsqueeze(1) + path.clamp(min=0).to(dtype=prefix.dtype)
        present = path >= 0
        valid = present & (live_req & self._active[:num_reqs]).unsqueeze(1)
        max_pos = self.runner.max_model_len - 1
        in_range = (src_pos >= 0) & (src_pos <= max_pos) & (dst_pos >= 0) & (dst_pos <= max_pos)
        valid &= in_range
        dst_pos = dst_pos.clamp(0, max_pos)
        src_pos = src_pos.clamp(0, max_pos)
        req_f = safe_idx.unsqueeze(1).expand(num_reqs, self.spec_len)
        moves = []
        for caches, block_table, block_size, gathers in self._groups:
            max_block = block_table.shape[1] - 1
            src_bi = torch.div(src_pos, block_size, rounding_mode="floor")
            dst_bi = torch.div(dst_pos, block_size, rounding_mode="floor")
            columns_valid = (src_bi <= max_block) & (dst_bi <= max_block)
            src_block = block_table[req_f, src_bi.clamp(max=max_block)]
            dst_block = block_table[req_f, dst_bi.clamp(max=max_block)]
            src_flat = (src_block * block_size + src_pos % block_size).to(dtype=torch.long).reshape(-1)
            dst_flat = (dst_block * block_size + dst_pos % block_size).to(dtype=torch.long).reshape(-1)
            # Layers in one block-table group share the device movement plan.
            # Capacity is a host shape; payload width/dtype do not affect slots.
            indices = {}
            for cache, gather in zip(caches, gathers):
                capacity = cache.shape[0]
                if capacity not in indices:
                    pages_valid = (src_block > 0) & (dst_block > 0) & (src_block < capacity) & (dst_block < capacity)
                    keep = (valid & columns_valid & pages_valid).reshape(-1)
                    safe_src = src_flat.clamp(0, capacity * block_size - 1)
                    source = torch.stack((safe_src // block_size, safe_src % block_size), dim=1)
                    destination = torch.stack((dst_flat // block_size, dst_flat % block_size), dim=1)
                    destination = torch.where(keep.unsqueeze(1), destination, -1)
                    indices[capacity] = source, destination
                source, destination = indices[capacity]
                moves.append(TreeRowMove(cache, source, destination, gather[:nslot]))
        relocate_tree_rows(moves, deduplicate=True)
