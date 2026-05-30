import inspect

import torch
import vllm.model_executor.layers.mamba.ops.ssd_chunk_scan as _chunk_scan
import vllm.model_executor.layers.mamba.ops.ssd_combined as _ssd_combined
import vllm.model_executor.layers.mamba.ops.ssd_state_passing as _state_passing

_ORIGINAL_STATE_PASSING_FWD = _state_passing._state_passing_fwd
_STATE_PASSING_PARAMS = set(inspect.signature(_ORIGINAL_STATE_PASSING_FWD).parameters)


def _resolve_accelerator_index(device_or_index):
    if isinstance(device_or_index, torch.device):
        if device_or_index.index is not None:
            return int(device_or_index.index)
        current_device_index = getattr(torch.accelerator, "current_device_index", None)
        if current_device_index is not None:
            return int(current_device_index())
        return 0
    return int(device_or_index)


def _chunk_scan_fwd(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    out,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    assert seq_idx is not None, "this implementation requires seq_idx"

    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (seqlen, ngroups, dstate)
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if z is not None:
        assert z.shape == x.shape
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert states.shape == (nchunks, nheads, headdim, dstate)
    assert seq_idx.shape == (nchunks,)

    grid = lambda META: (
        _chunk_scan.triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * _chunk_scan.triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    initial_states_ptr = initial_states
    initial_states_strides = (
        (
            initial_states.stride(0),
            initial_states.stride(1),
            initial_states.stride(2),
            initial_states.stride(3),
        )
        if initial_states is not None
        else (0, 0, 0, 0)
    )
    if initial_states_ptr is None:
        # Triton-Ascend still inspects pointer types in constexpr-false branches.
        initial_states_ptr = states.new_empty((1, 1, 1, 1))

    _chunk_scan._chunk_scan_fwd_kernel[grid](
        cb_ptr=cb,
        x_ptr=x,
        z_ptr=z,
        out_ptr=out,
        dt_ptr=dt,
        dA_cumsum_ptr=dA_cumsum,
        seq_idx_ptr=seq_idx,
        C_ptr=C,
        states_ptr=states,
        D_ptr=D,
        initstates_ptr=initial_states_ptr,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        chunk_size=chunk_size,
        hdim=headdim,
        dstate=dstate,
        seqlen=seqlen,
        nheads_ngroups_ratio=nheads // ngroups,
        stride_cb_chunk=cb.stride(0),
        stride_cb_head=cb.stride(1),
        stride_cb_csize_m=cb.stride(2),
        stride_cb_csize_k=cb.stride(3),
        stride_x_seqlen=x.stride(0),
        stride_x_head=x.stride(1),
        stride_x_hdim=x.stride(2),
        stride_z_seqlen=z_strides[0],
        stride_z_head=z_strides[1],
        stride_z_hdim=z_strides[2],
        stride_out_seqlen=out.stride(0),
        stride_out_head=out.stride(1),
        stride_out_hdim=out.stride(2),
        stride_dt_chunk=dt.stride(1),
        stride_dt_head=dt.stride(0),
        stride_dt_csize=dt.stride(2),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        stride_seq_idx_chunk=seq_idx.stride(0),
        stride_C_seqlen=C.stride(0),
        stride_C_head=C.stride(1),
        stride_C_dstate=C.stride(2),
        stride_states_chunk=states.stride(0),
        stride_states_head=states.stride(1),
        stride_states_hdim=states.stride(2),
        stride_states_dstate=states.stride(3),
        stride_init_states_batch=initial_states_strides[0],
        stride_init_states_head=initial_states_strides[1],
        stride_init_states_hdim=initial_states_strides[2],
        stride_init_states_dstate=initial_states_strides[3],
        stride_D_head=D.stride(0) if D is not None else 0,
        IS_CAUSAL=True,
        HAS_D=D is not None,
        D_HAS_HDIM=D.dim() == 2 if D is not None else True,
        HAS_Z=z is not None,
        BLOCK_SIZE_DSTATE=max(_chunk_scan.triton.next_power_of_2(dstate), 16),
        IS_TRITON_22=_chunk_scan.TRITON_22,
        HAS_INITSTATES=initial_states is not None,
    )
    return


def _state_passing_fwd(
    states,
    dA_cumsum,
    *args,
    initial_states=None,
    out_dtype=None,
    **kwargs,
):
    if "last_chunk_indices" in _STATE_PASSING_PARAMS:
        last_chunk_indices = kwargs.pop("last_chunk_indices", None)
        if last_chunk_indices is None:
            last_chunk_indices = args[0]
        return _state_passing_fwd_last_chunk_indices(
            states,
            dA_cumsum,
            last_chunk_indices,
            initial_states=initial_states,
            out_dtype=out_dtype,
        )

    cu_chunk_seqlens = kwargs.pop("cu_chunk_seqlens", None)
    seq_idx = kwargs.pop("seq_idx", None)
    if len(args) >= 1 and cu_chunk_seqlens is None:
        cu_chunk_seqlens = args[0]
    if len(args) >= 2 and seq_idx is None:
        seq_idx = args[1]
    return _state_passing_fwd_seq_idx(
        states,
        dA_cumsum,
        cu_chunk_seqlens,
        seq_idx,
        initial_states=initial_states,
        out_dtype=out_dtype,
    )


def _state_passing_fwd_last_chunk_indices(
    states,
    dA_cumsum,
    last_chunk_indices,
    initial_states=None,
    out_dtype=None,
):
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    batch = last_chunk_indices.shape[0]
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((nchunks, nheads, dim), device=states.device, dtype=out_dtype)

    initial_states_ptr = initial_states
    initial_states_strides = (
        (initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
        if initial_states is not None
        else (0, 0, 0)
    )
    if initial_states_ptr is None:
        # Match _chunk_scan_fwd: keep HAS_INITSTATES false while giving Ascend a typed ptr.
        initial_states_ptr = states.new_empty((1, 1, 1))

    grid = lambda META: (_state_passing.triton.cdiv(dim, META["BLOCK_SIZE"]), batch, nheads)
    with torch.accelerator.device_index(_resolve_accelerator_index(states.device)):
        _state_passing._state_passing_fwd_kernel[grid](
            states_ptr=states,
            out_ptr=out,
            dA_cs_ptr=dA_cumsum,
            initstates_ptr=initial_states_ptr,
            last_chunk_indices_ptr=last_chunk_indices,
            dim=dim,
            chunk_size=chunk_size,
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_dim=states.stride(2),
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_dim=out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_initstates_batch=initial_states_strides[0],
            stride_initstates_head=initial_states_strides[1],
            stride_initstates_dim=initial_states_strides[2],
            HAS_INITSTATES=initial_states is not None,
        )
    return out


def _state_passing_fwd_seq_idx(
    states,
    dA_cumsum,
    cu_chunk_seqlens,
    seq_idx,
    initial_states=None,
    out_dtype=None,
):
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert seq_idx is not None, "this implementation requires seq_idx"
    seqlen = seq_idx.shape[-1]
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((nchunks, nheads, dim), device=states.device, dtype=out_dtype)

    initial_states_ptr = initial_states
    initial_states_strides = (
        (initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
        if initial_states is not None
        else (0, 0, 0)
    )
    if initial_states_ptr is None:
        # Triton-Ascend still inspects pointer types in constexpr-false branches.
        initial_states_ptr = states.new_empty((1, 1, 1))

    grid = lambda META: (_state_passing.triton.cdiv(dim, META["BLOCK_SIZE"]), nheads)
    with torch.accelerator.device_index(_resolve_accelerator_index(states.device)):
        _state_passing._state_passing_fwd_kernel[grid](
            states_ptr=states,
            out_ptr=out,
            dA_cs_ptr=dA_cumsum,
            initstates_ptr=initial_states_ptr,
            seq_idx_ptr=seq_idx,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            dim=dim,
            nchunks=nchunks,
            seqlen=seqlen,
            chunk_size=chunk_size,
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_dim=states.stride(2),
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_dim=out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_initstates_batch=initial_states_strides[0],
            stride_initstates_head=initial_states_strides[1],
            stride_initstates_dim=initial_states_strides[2],
            stride_seq_idx_chunk=seq_idx.stride(0),
            HAS_INITSTATES=initial_states is not None,
        )
    return out


_chunk_scan._chunk_scan_fwd = _chunk_scan_fwd
_state_passing._state_passing_fwd = _state_passing_fwd
_ssd_combined._chunk_scan_fwd = _chunk_scan_fwd
_ssd_combined._state_passing_fwd = _state_passing_fwd
