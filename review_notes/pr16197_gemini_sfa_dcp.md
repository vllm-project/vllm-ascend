PR #16197 Gemini review evidence, SFA DCP indexer

P2P peer rank disposition: P2P_FALSE_POSITIVE

- Runtime pins in pyproject.toml are torch==2.10.0 and torch-npu==2.10.0.post4.
- PyTorch v2.10.0 `P2POp` keeps positional `peer` as a global rank and derives `group_peer` from it.
- PyTorch v2.10.0 `batch_isend_irecv` calls `isend`/`irecv` with `group_dst`/`group_src` from `P2POp.group_peer`.
- Cached torch_npu-2.10.0.post4 wheel `torch_npu/distributed/distributed_c10d.py` converts `p2p_op.peer` with `get_group_rank(group, p2p_op.peer)` for the non-coalesced NPU multi-PG path, and uses `op.group_peer` for coalesced NPU calls.
- Therefore the current indexer code path `peer = dcp_group.ranks[peer_in_group]` followed by `dist.P2POp(..., peer, group=dcp_group.device_group)` is already supplying the expected global peer rank for this pinned runtime.
