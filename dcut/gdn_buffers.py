# SPDX-License-Identifier: Apache-2.0
"""GDN metadata buffers used by eager and PIECEWISE GDN execution.

The legacy helpers keep their shared ASL/NAT layout for the retired
``patch_gdn.py`` path. The active v0.23 capture path owns batch-level fixed
buffers per model instance and padded token bucket. State indices are shared
by layer prefixes that reference the same attention metadata object, while
different metadata groups remain isolated. This keeps graph input addresses
stable without repeating the same metadata update for every GDN layer.
"""
from __future__ import annotations

import os

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from .globals import logger, _dcut_gdn_static

ENV_GDN_SHARED_STATIC = "VLLM_DCUT_GDN_SHARED_STATIC"


def _dcut_use_shared_gdn_static() -> bool:
    return os.environ.get(ENV_GDN_SHARED_STATIC, "1").lower() not in (
        "0", "false", "no"
    )


def _dcut_gdn_static_key(prefix, num_tokens, kind):
    # ASL/NAT are batch-level values shared by all Qwen3.5 GDN layers.
    # SSI comes from the per-layer metadata's state-index tuple and must stay
    # prefix-local, matching the original full-decode path.
    if kind in ("spec_asl_nat", "nonspec_asl") and _dcut_use_shared_gdn_static():
        owner = "__shared__"
    else:
        owner = prefix
    return (owner, num_tokens, kind)


def _to_int_tuple(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if hasattr(value, "tolist"):
        return tuple(int(v) for v in value.tolist())
    return (int(value),)


def _spec_host_args(meta) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Read spec conv1d host args from CPU metadata (no NPU sync)."""
    fallback_meta = getattr(meta, "spec_decode_fallback_meta", None)
    if fallback_meta is not None:
        conv_meta = fallback_meta.spec_causal_conv1d
        return (
            _to_int_tuple(conv_meta.query_start_loc_cpu),
            _to_int_tuple(conv_meta.cache_indices_cpu),
            _to_int_tuple(conv_meta.num_accepted_tokens_cpu),
        )
    # Fallback: .tolist() on device tensors causes NPU sync — avoid on hot path.
    return (
        _to_int_tuple(meta.spec_query_start_loc),
        _to_int_tuple(meta.spec_state_indices_tensor.reshape(-1)),
        _to_int_tuple(meta.num_accepted_tokens),
    )


def _nonspec_host_args(meta) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Read non-spec conv1d host args from CPU metadata (no NPU sync)."""
    fallback_meta = getattr(meta, "non_spec_decode_fallback_meta", None)
    if fallback_meta is not None:
        conv_meta = fallback_meta.causal_conv1d
        return (
            _to_int_tuple(conv_meta.query_start_loc_cpu),
            _to_int_tuple(conv_meta.cache_indices_cpu),
        )
    return (
        _to_int_tuple(meta.non_spec_query_start_loc),
        _to_int_tuple(meta.non_spec_state_indices_tensor),
    )


def _dcut_alloc_gdn_spec_bufs(prefix, num_tokens, spec_state_indices_tensor, device):
    """Allocate shared ASL/NAT + per-layer SSI buffers for spec decode path."""
    b_cap = spec_state_indices_tensor.size(0)
    nsp1 = spec_state_indices_tensor.size(1)  # num_spec + 1
    t_cap = b_cap * nsp1

    # Shared ASL/NAT (all layers share same batch composition)
    shared_key = _dcut_gdn_static_key(prefix, num_tokens, "spec_asl_nat")
    if shared_key not in _dcut_gdn_static:
        _dcut_gdn_static[shared_key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "nat": torch.zeros(b_cap, dtype=torch.int32, device=device),
            "asl_cpu": torch.zeros(b_cap + 1, dtype=torch.int32, device="cpu"),
            "nat_cpu": torch.zeros(b_cap, dtype=torch.int32, device="cpu"),
            "b_cap": b_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN shared spec ASL/NAT num_tokens=%d b_cap=%d",
            num_tokens, b_cap)

    # Per-layer SSI
    ssi_key = _dcut_gdn_static_key(prefix, num_tokens, "spec_ssi")
    if ssi_key not in _dcut_gdn_static:
        _dcut_gdn_static[ssi_key] = {
            "ssi": torch.full((t_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            "col_idx": torch.arange(nsp1, device=device),
            "b_cap": b_cap,
            "nsp1": nsp1,
            "t_cap": t_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN spec SSI prefix=%s num_tokens=%d b_cap=%d t_cap=%d",
            prefix, num_tokens, b_cap, t_cap)

    # Return combined dict for convenience
    return {**_dcut_gdn_static[shared_key], **_dcut_gdn_static[ssi_key]}


def _dcut_fill_gdn_spec_bufs(prefix, num_tokens, meta, device,
                              fill_shared_asl_nat=True):
    """Fill ASL/SSI/NAT buffers before graph replay.

    Optimization: ASL/NAT built on CPU from conv1d host args, one copy to GPU.
    SSI still uses NPU ops (source is device tensor, unavoidable).
    """
    spec_state_indices_tensor = meta.spec_state_indices_tensor
    num_spec_decodes = meta.num_spec_decodes
    spec_query_start_loc = meta.spec_query_start_loc
    num_accepted_tokens = meta.num_accepted_tokens

    bufs = _dcut_alloc_gdn_spec_bufs(
        prefix, num_tokens, spec_state_indices_tensor, device)

    # --- Shared ASL/NAT: CPU build + one GPU copy ---
    if fill_shared_asl_nat:
        asl_cpu = bufs["asl_cpu"]
        nat_cpu = bufs["nat_cpu"]
        asl_cpu.zero_()
        nat_cpu.zero_()

        if num_spec_decodes > 0:
            # Try CPU host args first (no NPU sync)
            qsl_host, _, nat_host = _spec_host_args(meta)

            if qsl_host and nat_host:
                # ASL: diff of cumulative query_start_loc
                n = min(num_spec_decodes, len(qsl_host) - 1)
                for i in range(n):
                    asl_cpu[i + 1] = qsl_host[i + 1] - qsl_host[i]
                # NAT: from host args, clamped to segment length
                n_nat = min(num_spec_decodes, len(nat_host))
                for i in range(n_nat):
                    val = nat_host[i]
                    if i + 1 < len(qsl_host):
                        seg_len = qsl_host[i + 1] - qsl_host[i]
                        val = min(val, seg_len)
                    nat_cpu[i] = val
            else:
                # Fallback: build from NPU tensors (causes sync, but correct)
                cu = spec_query_start_loc[:num_spec_decodes + 1]
                per_seq_lens = cu[1:] - cu[:-1]
                asl_cpu[1:num_spec_decodes + 1].copy_(per_seq_lens.cpu())
                clamped = torch.minimum(
                    num_accepted_tokens[:num_spec_decodes].to(torch.int32),
                    per_seq_lens.to(torch.int32)
                )
                nat_cpu[:num_spec_decodes].copy_(clamped.cpu())

        # One copy CPU -> GPU
        bufs["asl"].copy_(asl_cpu, non_blocking=True)
        bufs["nat"].copy_(nat_cpu, non_blocking=True)

    # --- Per-layer SSI: NPU path (unavoidable, source is device tensor) ---
    ssi = bufs["ssi"]
    ssi.fill_(PAD_SLOT_ID)

    if num_spec_decodes > 0:
        cu = spec_query_start_loc[:num_spec_decodes + 1]
        per_seq_lens = cu[1:] - cu[:-1]
        col_idx = bufs["col_idx"]
        mask = col_idx.unsqueeze(0) < per_seq_lens.unsqueeze(1)
        real = spec_state_indices_tensor[:num_spec_decodes][mask]
        ssi[:real.size(0)].copy_(real)

    return bufs


def _dcut_alloc_gdn_nonspec_bufs(prefix, num_tokens,
                                  non_spec_state_indices_tensor, device):
    """Allocate shared ASL + per-layer SSI buffers for non-spec decode path."""
    b_cap = non_spec_state_indices_tensor.size(0)

    # Shared ASL
    shared_key = _dcut_gdn_static_key(prefix, num_tokens, "nonspec_asl")
    if shared_key not in _dcut_gdn_static:
        _dcut_gdn_static[shared_key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "asl_cpu": torch.zeros(b_cap + 1, dtype=torch.int32, device="cpu"),
            "b_cap": b_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN shared nonspec ASL num_tokens=%d b_cap=%d",
            num_tokens, b_cap)

    # Per-layer SSI
    ssi_key = _dcut_gdn_static_key(prefix, num_tokens, "nonspec_ssi")
    if ssi_key not in _dcut_gdn_static:
        _dcut_gdn_static[ssi_key] = {
            "ssi": torch.full((b_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        }
        logger.debug(
            "D-Cut: alloc GDN nonspec SSI prefix=%s num_tokens=%d b_cap=%d",
            prefix, num_tokens, b_cap)

    return {**_dcut_gdn_static[shared_key], **_dcut_gdn_static[ssi_key]}


def _dcut_fill_gdn_nonspec_bufs(prefix, num_tokens, meta, device,
                                 fill_shared_asl=True):
    """Fill ASL/SSI buffers for non-spec decode before graph replay."""
    non_spec_query_start_loc = meta.non_spec_query_start_loc
    non_spec_state_indices_tensor = meta.non_spec_state_indices_tensor
    num_decodes = meta.num_decodes

    bufs = _dcut_alloc_gdn_nonspec_bufs(
        prefix, num_tokens, non_spec_state_indices_tensor, device)

    # --- Shared ASL: CPU build + one GPU copy ---
    if fill_shared_asl:
        asl_cpu = bufs["asl_cpu"]
        asl_cpu.zero_()

        if num_decodes > 0:
            qsl_host, _ = _nonspec_host_args(meta)
            if qsl_host:
                n = min(num_decodes, len(qsl_host) - 1)
                for i in range(n):
                    asl_cpu[i + 1] = qsl_host[i + 1] - qsl_host[i]
            else:
                cu = non_spec_query_start_loc[:num_decodes + 1]
                asl_cpu[1:num_decodes + 1].copy_((cu[1:] - cu[:-1]).cpu())

        bufs["asl"].copy_(asl_cpu, non_blocking=True)

    # --- Per-layer SSI: NPU path ---
    ssi = bufs["ssi"]
    ssi.fill_(PAD_SLOT_ID)
    if num_decodes > 0:
        ssi[:num_decodes].copy_(non_spec_state_indices_tensor[:num_decodes])

    return bufs


def _dcut_update_gdn_static(forward_context, num_tokens, GDNAttentionMetadata):
    """Update GDN static buffers from forward context's attn_metadata.

    Called from patched _model_forward before _orig_model_forward (i.e. before
    graph replay). ASL/NAT are filled once (shared), SSI per-layer.
    """
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None or not isinstance(attn_metadata, dict):
        return
    filled_shared_keys = set()
    for prefix, meta in attn_metadata.items():
        if not isinstance(meta, GDNAttentionMetadata):
            continue
        if meta.spec_sequence_masks is not None and meta.num_spec_decodes > 0:
            shared_key = _dcut_gdn_static_key(prefix, num_tokens, "spec_asl_nat")
            fill_shared = shared_key not in filled_shared_keys
            filled_shared_keys.add(shared_key)
            _dcut_fill_gdn_spec_bufs(
                prefix, num_tokens, meta, meta.spec_query_start_loc.device,
                fill_shared_asl_nat=fill_shared,
            )
        elif meta.num_decodes > 0:
            shared_key = _dcut_gdn_static_key(prefix, num_tokens, "nonspec_asl")
            fill_shared = shared_key not in filled_shared_keys
            filled_shared_keys.add(shared_key)
            _dcut_fill_gdn_nonspec_bufs(
                prefix, num_tokens, meta, meta.non_spec_query_start_loc.device,
                fill_shared_asl=fill_shared,
            )


def _dcut_prepare_gdn_eager_state(
    forward_context,
    GDNAttentionMetadata,
    *,
    initial_spec_rows=(),
) -> bool:
    """Build batch-level speculative state once for all eager GDN layers.

    ``query_start_loc`` and ``num_accepted_tokens`` describe the batch and
    are identical for every GDN layer. Keep state indices layer-local, but
    share these batch-level tensors for the duration of one model forward.
    """
    forward_context._dcut_gdn_eager_spec_state = None
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict):
        return False

    spec_items = [
        meta
        for meta in attn_metadata.values()
        if (
            isinstance(meta, GDNAttentionMetadata)
            and meta.spec_sequence_masks is not None
            and int(meta.num_spec_decodes) > 0
            and meta.spec_decode_metadata is not None
        )
    ]
    if not spec_items:
        return False

    reference = spec_items[0]
    num_spec_decodes = int(reference.num_spec_decodes)
    spec_decode_metadata = reference.spec_decode_metadata
    conv_metadata = spec_decode_metadata.spec_causal_conv1d
    num_accepted_tokens = conv_metadata.num_accepted_tokens
    query_start_loc = conv_metadata.query_start_loc
    num_conv_requests = int(conv_metadata.cache_indices.shape[0])
    if (
        num_accepted_tokens is None
        or query_start_loc is None
        or num_accepted_tokens.numel() < num_conv_requests
    ):
        return False

    # Only share metadata when every GDN layer describes the same batch
    # shape. Values are produced from the same scheduler output; state indices
    # intentionally remain on each layer's metadata object.
    for meta in spec_items[1:]:
        layer_spec_metadata = meta.spec_decode_metadata
        layer_conv_metadata = layer_spec_metadata.spec_causal_conv1d
        if (
            int(meta.num_spec_decodes) != num_spec_decodes
            or tuple(layer_conv_metadata.num_accepted_tokens.shape)
            != tuple(num_accepted_tokens.shape)
            or int(layer_conv_metadata.cache_indices.shape[0])
            != num_conv_requests
            or tuple(layer_conv_metadata.query_start_loc.shape)
            != tuple(query_start_loc.shape)
        ):
            return False

    accepted_tokens_int32 = num_accepted_tokens
    if accepted_tokens_int32.dtype != torch.int32:
        accepted_tokens_int32 = accepted_tokens_int32.to(torch.int32)

    initial_spec_rows = tuple(int(row) for row in initial_spec_rows)
    if initial_spec_rows:
        if any(
            row < 0 or row >= num_spec_decodes
            for row in initial_spec_rows
        ):
            raise RuntimeError(
                "D-Cut eager handoff row is outside the compact speculative "
                f"batch: rows={initial_spec_rows}, requests={num_spec_decodes}"
            )
        # The graph-unsafe fallback must preserve the same first-step state
        # selector as graph replay. Clone only for a handoff batch so ordinary
        # eager decode keeps the allocation-free shared metadata path.
        accepted_tokens_int32 = accepted_tokens_int32.clone()
        if initial_spec_rows == tuple(range(num_spec_decodes)):
            accepted_tokens_int32[:num_spec_decodes].fill_(1)
        else:
            row_indices = torch.tensor(
                initial_spec_rows,
                dtype=torch.long,
                device=accepted_tokens_int32.device,
            )
            accepted_tokens_int32.index_fill_(0, row_indices, 1)

    forward_context._dcut_gdn_eager_spec_state = {
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": accepted_tokens_int32,
    }
    return True


def _dcut_gdn_piecewise_spec_key(forward_context, prefix, num_tokens):
    """Return a prefix lookup key that does not alias model instances."""
    model_instance = getattr(forward_context, "model_instance", None)
    return (
        id(model_instance),
        prefix,
        num_tokens,
        "v023_piecewise_spec_ssi",
    )


def _dcut_gdn_piecewise_shared_key(forward_context, num_tokens):
    """Return the key for batch metadata shared by all GDN graph segments."""
    model_instance = getattr(forward_context, "model_instance", None)
    return (
        id(model_instance),
        num_tokens,
        "v023_piecewise_spec_shared",
    )


def _dcut_alloc_gdn_piecewise_spec_bufs(
    forward_context,
    prefixes,
    num_tokens,
    state_indices,
    max_num_seqs,
):
    """Allocate fixed-shape inputs consumed by a PIECEWISE GDN graph.

    PIECEWISE ACLGraph keys contain the padded token count, but not the live
    number of speculative requests, so its caller uses the scheduler request
    capacity. Ragged FULL uses the maximum request count reachable by that token
    bucket. In both modes, the capacity is fixed for every graph of a given
    token size so different live request compositions can safely replay it.
    Batch-level inputs are shared across GDN layers. Prefixes backed by one
    attention metadata object also share state indices; separate metadata
    groups keep separate state-index buffers.
    """
    if state_indices.ndim != 2:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN requires 2-D per-request state indices, "
            f"got shape={tuple(state_indices.shape)}"
        )

    if not prefixes:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN buffer group has no layer prefixes"
        )

    layer_keys = [
        _dcut_gdn_piecewise_spec_key(
            forward_context, prefix, num_tokens
        )
        for prefix in prefixes
    ]
    state_index_stride = state_indices.shape[1]
    expected_shape = (max_num_seqs, state_index_stride)
    existing_bufs = [
        _dcut_gdn_static[key]
        for key in layer_keys
        if key in _dcut_gdn_static
    ]
    bufs = existing_bufs[0] if existing_bufs else None
    if bufs is not None:
        if any(existing is not bufs for existing in existing_bufs[1:]):
            raise RuntimeError(
                "D-Cut PIECEWISE GDN prefixes resolved to different "
                f"state-index buffers: prefixes={tuple(prefixes)}"
            )
        if tuple(bufs["ssi"].shape) != expected_shape:
            raise RuntimeError(
                "D-Cut PIECEWISE GDN buffer shape changed for an existing "
                f"graph key: expected={expected_shape}, "
                f"actual={tuple(bufs['ssi'].shape)}"
            )
        for key in layer_keys:
            _dcut_gdn_static[key] = bufs
        return bufs

    shared_key = _dcut_gdn_piecewise_shared_key(
        forward_context, num_tokens
    )
    shared_bufs = _dcut_gdn_static.get(shared_key)
    if shared_bufs is None:
        device = state_indices.device
        shared_bufs = {
            "qsl": torch.zeros(
                max_num_seqs + 1, dtype=torch.int32, device=device
            ),
            "nat": torch.zeros(
                max_num_seqs, dtype=torch.int32, device=device
            ),
        }
        _dcut_gdn_static[shared_key] = shared_bufs
        logger.info(
            "D-Cut: allocated shared v0.23 PIECEWISE GDN buffers "
            "num_tokens=%d max_num_seqs=%d",
            num_tokens,
            max_num_seqs,
        )
    elif tuple(shared_bufs["qsl"].shape) != (max_num_seqs + 1,):
        raise RuntimeError(
            "D-Cut PIECEWISE GDN shared buffer shape changed for an "
            f"existing graph key: expected={(max_num_seqs + 1,)}, "
            f"actual={tuple(shared_bufs['qsl'].shape)}"
        )

    bufs = {
        **shared_bufs,
        "ssi": torch.full(
            expected_shape,
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=state_indices.device,
        ),
    }
    for key in layer_keys:
        _dcut_gdn_static[key] = bufs
    logger.info(
        "D-Cut: allocated metadata-group v0.23 PIECEWISE GDN state "
        "indices group_prefix=%s group_layers=%d num_tokens=%d "
        "max_num_seqs=%d stride=%d",
        prefixes[0],
        len(prefixes),
        num_tokens,
        max_num_seqs,
        state_index_stride,
    )
    return bufs


def _dcut_fill_gdn_piecewise_spec_bufs(
    forward_context,
    prefixes,
    num_tokens,
    meta,
    max_num_seqs,
    *,
    fill_shared_batch,
    clear_unused_rows,
    initial_spec_rows=(),
):
    """Refresh shared batch inputs once and layer-local state indices."""
    num_spec_decodes = int(meta.num_spec_decodes)
    if num_spec_decodes <= 0 or num_spec_decodes > max_num_seqs:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN speculative request count is outside "
            f"the configured capacity: requests={num_spec_decodes}, "
            f"capacity={max_num_seqs}"
        )

    spec_decode_metadata = meta.spec_decode_metadata
    conv_meta = spec_decode_metadata.spec_causal_conv1d
    state_indices = meta.spec_state_indices_tensor
    if state_indices is None:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN is missing spec_state_indices_tensor"
        )

    bufs = _dcut_alloc_gdn_piecewise_spec_bufs(
        forward_context,
        prefixes,
        num_tokens,
        state_indices,
        max_num_seqs,
    )

    if fill_shared_batch:
        qsl = bufs["qsl"]
        if clear_unused_rows:
            # FULL replay reuses these addresses across unrelated request
            # lifetimes. Rebuild the complete fixed-capacity buffer before
            # publishing the active prefix, so neither a shrinking batch nor
            # an interrupted prior update can leave a valid-looking boundary.
            terminal = conv_meta.query_start_loc[
                num_spec_decodes : num_spec_decodes + 1
            ]
            qsl.copy_(terminal.expand_as(qsl), non_blocking=True)
        qsl[: num_spec_decodes + 1].copy_(
            conv_meta.query_start_loc[: num_spec_decodes + 1],
            non_blocking=True,
        )
        if not clear_unused_rows:
            qsl_tail = qsl[num_spec_decodes + 1 :]
            if qsl_tail.numel() > 0:
                qsl_tail.copy_(
                    qsl[num_spec_decodes].expand_as(qsl_tail),
                    non_blocking=True,
                )

        nat = bufs["nat"]
        if clear_unused_rows:
            # One is vLLM's valid default state position. Inactive rows have a
            # zero-length QSL segment, but initializing NAT as well makes the
            # fixed graph input deterministic before the active copy.
            nat.fill_(1)
        accepted_tokens = conv_meta.num_accepted_tokens[:num_spec_decodes]
        if accepted_tokens.dtype != torch.int32:
            accepted_tokens = accepted_tokens.to(torch.int32)
        # This selects the state produced by the *previous* verifier step. Its
        # position is independent of the number of tokens retained by D-Cut for
        # the current step. Clamping it to the current segment length makes a
        # shrinking verifier read an older conv/recurrent state than eager mode.
        nat[:num_spec_decodes].copy_(
            accepted_tokens,
            non_blocking=True,
        )
        # A zero-draft KV handoff is the consumer's first verifier step. Its
        # transferred state is h(N - 1), so the state selector must be one.
        # FULL replay reuses fixed metadata addresses across request lifetimes;
        # normalize only these explicit first-step rows so a recycled batch
        # slot cannot select an accepted-token offset from the previous request.
        initial_spec_rows = tuple(int(row) for row in initial_spec_rows)
        if initial_spec_rows:
            if any(
                row < 0 or row >= num_spec_decodes
                for row in initial_spec_rows
            ):
                raise RuntimeError(
                    "D-Cut initial handoff row is outside the compact "
                    f"speculative batch: rows={initial_spec_rows}, "
                    f"requests={num_spec_decodes}"
                )
            if initial_spec_rows == tuple(range(num_spec_decodes)):
                nat[:num_spec_decodes].fill_(1)
            else:
                row_key = (
                    id(getattr(forward_context, "model_instance", None)),
                    num_tokens,
                    initial_spec_rows,
                    "v023_initial_spec_rows",
                )
                row_indices = _dcut_gdn_static.get(row_key)
                if row_indices is None:
                    row_indices = torch.tensor(
                        initial_spec_rows,
                        dtype=torch.long,
                        device=nat.device,
                    )
                    _dcut_gdn_static[row_key] = row_indices
                nat.index_fill_(0, row_indices, 1)

    ssi = bufs["ssi"]
    if clear_unused_rows:
        # Reinitialize the complete request axis. The custom Conv1D operator
        # rejects PAD_SLOT_ID, while the recurrent operator skips the matching
        # zero-length QSL rows before reading SSI.
        ssi.fill_(PAD_SLOT_ID)
    ssi[:num_spec_decodes].copy_(
        state_indices[:num_spec_decodes],
        non_blocking=True,
    )
    return bufs


def _dcut_graph_capture_qsl(
    num_tokens: int,
    spec_query_len: int,
    max_num_seqs: int,
) -> tuple[int, ...]:
    """Build a synthetic pure-spec layout for one graph token bucket."""
    if num_tokens <= 0 or spec_query_len <= 0 or max_num_seqs <= 0:
        return ()
    num_spec_decodes = (
        num_tokens + spec_query_len - 1
    ) // spec_query_len
    if num_spec_decodes > max_num_seqs:
        return ()
    return tuple(
        min(row * spec_query_len, num_tokens)
        for row in range(num_spec_decodes + 1)
    )


def _dcut_prepare_gdn_graph_capture(
    forward_context,
    num_tokens,
    GDNAttentionMetadata,
    max_num_seqs,
    spec_query_len,
):
    """Prepare fixed GDN inputs while vLLM captures a non-uniform graph key.

    PIECEWISE deliberately reuses one key for mixed and uniform batches, while
    ragged FULL fixes request capacity independently of the live batch size.
    Their stock dummy metadata therefore need not describe the pure-spec batch
    that D-Cut later replays. Build only the GDN inputs as a synthetic
    pure-spec batch; attention and the rest of the model retain stock capture
    metadata.
    """
    capture_qsl = _dcut_graph_capture_qsl(
        int(num_tokens),
        int(spec_query_len),
        int(max_num_seqs),
    )
    if not capture_qsl:
        return False
    num_spec_decodes = len(capture_qsl) - 1

    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict):
        return False

    gdn_items = [
        (prefix, meta)
        for prefix, meta in attn_metadata.items()
        if isinstance(meta, GDNAttentionMetadata)
    ]
    if not gdn_items:
        return False

    metadata_groups = {}
    for prefix, meta in gdn_items:
        group = metadata_groups.setdefault(
            id(meta), {"meta": meta, "prefixes": []}
        )
        group["prefixes"].append(prefix)

    for group in metadata_groups.values():
        meta = group["meta"]
        spec_indices = getattr(meta, "spec_state_indices_tensor", None)
        non_spec_indices = getattr(
            meta, "non_spec_state_indices_tensor", None
        )
        state_indices = None
        if (
            spec_indices is not None
            and spec_indices.ndim == 2
            and int(spec_indices.shape[0]) >= num_spec_decodes
            and int(spec_indices.shape[1]) >= spec_query_len
        ):
            state_indices = spec_indices[:, :spec_query_len]
        elif (
            non_spec_indices is not None
            and non_spec_indices.ndim == 1
            and int(non_spec_indices.shape[0]) >= num_spec_decodes
        ):
            # Stock mixed PIECEWISE dummy metadata exposes one state slot per
            # request. The real speculative path exposes num_spec + 1 slots.
            # Repeat the valid dummy slot only to establish the captured
            # tensor geometry; runtime replay overwrites all active rows with
            # the real speculative state-index matrix.
            state_indices = non_spec_indices.unsqueeze(1).expand(
                -1, spec_query_len
            )
        elif (
            non_spec_indices is not None
            and non_spec_indices.ndim == 2
            and int(non_spec_indices.shape[0]) >= num_spec_decodes
        ):
            if int(non_spec_indices.shape[1]) == spec_query_len:
                state_indices = non_spec_indices
            elif int(non_spec_indices.shape[1]) == 1:
                state_indices = non_spec_indices.expand(
                    -1, spec_query_len
                )
        if state_indices is None:
            return False
        group["state_indices"] = state_indices

    claimed_buffer_ids = set()
    for group in metadata_groups.values():
        existing_buffer_ids = set()
        for prefix in group["prefixes"]:
            key = _dcut_gdn_piecewise_spec_key(
                forward_context, prefix, num_tokens
            )
            if key in _dcut_gdn_static:
                existing_buffer_ids.add(id(_dcut_gdn_static[key]))
        if (
            len(existing_buffer_ids) > 1
            or not claimed_buffer_ids.isdisjoint(existing_buffer_ids)
        ):
            return False
        claimed_buffer_ids.update(existing_buffer_ids)

    for index, group in enumerate(metadata_groups.values()):
        state_indices = group["state_indices"]
        bufs = _dcut_alloc_gdn_piecewise_spec_bufs(
            forward_context,
            tuple(group["prefixes"]),
            num_tokens,
            state_indices,
            max_num_seqs,
        )
        if index == 0:
            qsl = bufs["qsl"]
            qsl.fill_(num_tokens)
            qsl[: num_spec_decodes + 1].copy_(
                torch.tensor(
                    capture_qsl,
                    dtype=torch.int32,
                    device=qsl.device,
                ),
                non_blocking=True,
            )
            bufs["nat"].fill_(1)

        ssi = bufs["ssi"]
        ssi.fill_(PAD_SLOT_ID)
        ssi[:num_spec_decodes].copy_(
            state_indices[:num_spec_decodes],
            non_blocking=True,
        )
    return True


def _dcut_prepare_gdn_piecewise_replay(
    forward_context,
    num_tokens,
    GDNAttentionMetadata,
    max_num_seqs,
    *,
    clear_unused_rows=False,
    initial_spec_rows=(),
):
    """Prepare fixed pure-spec GDN metadata, or reject it safely.

    The GDN custom op chooses prefill/decode/spec branches from Python
    metadata that is not part of the compiled graph signature. Only a pure
    speculative batch may enter the expanded graph. Other compositions retain
    the whole-core eager boundary while outer PIECEWISE stays active.
    """
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict):
        return False

    gdn_items = [
        (prefix, meta)
        for prefix, meta in attn_metadata.items()
        if isinstance(meta, GDNAttentionMetadata)
    ]
    if not gdn_items:
        return False

    # vLLM assigns the same metadata object to every layer prefix in one
    # attention group. Group by identity so QSL/NAT/SSI are refreshed once per
    # group while hybrid-cache groups remain independent.
    metadata_groups = {}
    for prefix, meta in gdn_items:
        group = metadata_groups.setdefault(
            id(meta), {"meta": meta, "prefixes": []}
        )
        group["prefixes"].append(prefix)

    for group in metadata_groups.values():
        meta = group["meta"]
        if (
            meta.spec_sequence_masks is None
            or int(meta.num_spec_decodes) <= 0
            or int(meta.num_prefills) != 0
            or int(meta.num_decodes) != 0
            or meta.spec_decode_metadata is None
        ):
            return False
        state_indices = meta.spec_state_indices_tensor
        if (
            state_indices is None
            or state_indices.ndim != 2
            or int(meta.num_spec_decodes) > max_num_seqs
        ):
            return False

    # A captured prefix must keep the same alias topology. Reject a runtime
    # attention-group change instead of overwriting a buffer captured for a
    # different metadata group.
    claimed_buffer_ids = set()
    for group in metadata_groups.values():
        existing_buffer_ids = set()
        for prefix in group["prefixes"]:
            key = _dcut_gdn_piecewise_spec_key(
                forward_context, prefix, num_tokens
            )
            if key in _dcut_gdn_static:
                existing_buffer_ids.add(id(_dcut_gdn_static[key]))
        if (
            len(existing_buffer_ids) > 1
            or not claimed_buffer_ids.isdisjoint(existing_buffer_ids)
        ):
            return False
        claimed_buffer_ids.update(existing_buffer_ids)

    for index, group in enumerate(metadata_groups.values()):
        _dcut_fill_gdn_piecewise_spec_bufs(
            forward_context,
            tuple(group["prefixes"]),
            num_tokens,
            group["meta"],
            max_num_seqs,
            fill_shared_batch=index == 0,
            clear_unused_rows=clear_unused_rows,
            initial_spec_rows=initial_spec_rows,
        )
    return True


def _dcut_get_gdn_piecewise_spec_bufs(
    forward_context,
    prefix,
    num_tokens,
):
    """Get buffers already prepared by the graph-external runner hook."""
    key = _dcut_gdn_piecewise_spec_key(
        forward_context, prefix, num_tokens
    )
    try:
        return _dcut_gdn_static[key]
    except KeyError as exc:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN buffers were not prepared before capture: "
            f"prefix={prefix}, num_tokens={num_tokens}"
        ) from exc
