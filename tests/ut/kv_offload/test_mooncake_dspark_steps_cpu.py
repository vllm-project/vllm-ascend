# SPDX-License-Identifier: Apache-2.0
"""Execute real connector address generation and CPU copies without an NPU."""

import ast
import ctypes
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def _method(path, class_name, method_name, **namespace):
    source = ROOT / "vllm_ascend" / "distributed" / "kv_transfer" / "kv_p2p" / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if getattr(node, "name", None) == method_name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[method_name]


class _Backend(Enum):
    MAMBA1 = 1
    MAMBA2 = 2
    GDN_ATTN = 3


def _cache(shape, dtype, fill):
    size = int(np.prod(shape))
    backing = np.full((4, size + 37), fill, dtype=dtype)
    view = np.lib.stride_tricks.as_strided(
        backing,
        shape=(4, *shape),
        strides=(backing.strides[0], *np.empty(shape, dtype=dtype).strides),
    )
    return backing, view


def _projection_shard(full, rank, tp):
    return np.concatenate([np.split(part, tp, axis=1)[rank] for part in np.split(full, 3, axis=1)], axis=1)


@pytest.mark.parametrize("p_steps,d_steps", [(1, 3), (3, 1), (3, 3)])
@pytest.mark.parametrize(
    "connector,layout,p_tp,d_tp",
    [
        ("v1", "SD", 8, 8),
        ("v1", "SD", 16, 8),
        ("v1", "DS", 8, 8),
        ("v1", "DS", 16, 8),
        ("v2", "SD", 8, 8),
        ("v2", "SD", 16, 8),
        ("v2", "SD", 4, 8),
    ],
)
def test_committed_conv_history_transfer_preserves_speculative_scratch(p_steps, d_steps, p_tp, d_tp, connector, layout):
    d_rank = 1
    heads, dim, history = 16, 2, 3
    d_width = heads * dim * 3 // d_tp
    p_width = heads * dim * 3 // p_tp
    d_shape = (history + d_steps, d_width)
    p_shape = (history + p_steps, p_width)
    if layout == "DS":
        d_shape, p_shape = d_shape[::-1], p_shape[::-1]
    d_backing, d_conv = _cache(d_shape, np.float16, -111)
    d_state_backing, d_state = _cache((heads // d_tp, dim, dim), np.float32, -222)
    before_conv = d_backing.copy()
    before_state = d_state_backing.copy()
    full_history = np.arange(history * heads * dim * 3, dtype=np.float16).reshape(history, -1)
    full_state = np.arange(heads * dim * dim, dtype=np.float32).reshape(heads, dim, dim)
    expected_conv = _projection_shard(full_history, d_rank, d_tp)
    expected_state = np.split(full_state, d_tp)[d_rank]
    linear_config = {"num_heads": heads, "head_dim": dim, "short_conv_kernel_size": 4}
    config = SimpleNamespace(hf_text_config=SimpleNamespace(linear_attn_config=linear_config))
    local_block, remote_block = 1, 2
    remote_ranks = (
        list(range(d_rank * (p_tp // d_tp), (d_rank + 1) * (p_tp // d_tp)))
        if p_tp >= d_tp
        else [d_rank // (d_tp // p_tp)]
    )
    for p_rank in remote_ranks:
        p_backing, p_conv = _cache(p_shape, np.float16, 1234)
        p_state_backing, p_state = _cache((heads // p_tp, dim, dim), np.float32, 5678)
        producer_history = _projection_shard(full_history, p_rank, p_tp)
        if layout == "SD":
            p_conv[remote_block, :history] = producer_history
        else:
            p_conv[remote_block, :, :history] = producer_history.T
        p_state[remote_block] = np.split(full_state, p_tp)[p_rank]
        p_before = p_backing.copy()
        p_state_before = p_state_backing.copy()
        local_bases = [d_conv.ctypes.data, d_state.ctypes.data]
        remote_bases = [p_conv.ctypes.data, p_state.ctypes.data]
        local_strides = [d_conv.strides[0], d_state.strides[0]]
        remote_strides = [p_conv.strides[0], p_state.strides[0]]
        local_lens = [d_conv[0].nbytes, d_state[0].nbytes]
        remote_lens = [p_conv[0].nbytes, p_state[0].nbytes]
        src, dst, sizes = [], [], []
        if connector == "v1":
            fn = _method(
                "mooncake_connector.py",
                "KVCacheRecvingThread",
                "_append_mamba_transfer_meta",
                is_conv_state_dim_first=lambda: layout == "DS",
            )
            thread = SimpleNamespace(tp_size=d_tp, vllm_config=SimpleNamespace(model_config=config))
            fn(
                thread,
                src,
                dst,
                sizes,
                group_spec={"shapes": [d_shape, d_state.shape[1:]], "dtype_sizes": [2, 4]},
                remote_group_spec={"shapes": [p_shape, p_state.shape[1:]], "dtype_sizes": [2, 4]},
                src_layer_base_addr=local_bases,
                dst_layer_base_addr=remote_bases,
                block_len=local_lens,
                block_stride=local_strides,
                remote_block_stride=remote_strides,
                remote_block_id=remote_block,
                local_block_id=local_block,
                tp_num_need_pulls=p_tp // d_tp,
                remote_tp_offset=p_rank % (p_tp // d_tp),
            )
        else:
            fn = _method(
                "mooncake/pull_worker.py",
                "MooncakePullRecvingThread",
                "_append_mamba_transfer_addresses",
                torch=torch,
                MambaAttentionBackendEnum=_Backend,
            )
            thread = SimpleNamespace(
                tp_size=d_tp,
                tp_rank=d_rank,
                model_config=config,
                layer_names=["model.layers.0.linear_attn"],
                kv_caches_base_addr=[local_bases],
                block_shapes=[[d_shape, d_state.shape[1:]]],
                block_lens=[local_lens],
                block_strides=[local_strides],
            )
            remote = SimpleNamespace(
                metadata_by_tp_rank={p_rank: SimpleNamespace(kv_caches_base_addr=[remote_bases])},
                block_shapes=[[p_shape, p_state.shape[1:]]],
                block_lens=[remote_lens],
                block_strides=[remote_strides],
            )
            spec = SimpleNamespace(dtypes=(torch.float16, torch.float32), mamba_type=_Backend.GDN_ATTN)
            fn(
                thread,
                spec,
                p_rank,
                p_tp,
                {(0, 0): [("request", [local_block], [remote_block])]},
                remote,
                src,
                dst,
                sizes,
            )

        # Mooncake calls these src/dst, but READ writes src (D) from dst (P).
        p_mask = np.zeros_like(p_backing, dtype=np.uint8)
        p_mask_view = np.lib.stride_tricks.as_strided(
            p_mask, shape=p_conv.shape, strides=tuple(s // 2 for s in p_conv.strides)
        )
        if layout == "SD":
            p_mask_view[remote_block, :history] = 1
        else:
            p_mask_view[remote_block, :, :history] = 1
        for local_addr, remote_addr, size in zip(src, dst, sizes):
            if p_conv.ctypes.data <= remote_addr < p_conv.ctypes.data + p_backing.nbytes:
                offset = remote_addr - p_conv.ctypes.data
                assert np.all(p_mask.ravel()[offset // 2 : (offset + size) // 2])
            else:
                assert p_state[remote_block].ctypes.data <= remote_addr
                assert remote_addr + size <= p_state[remote_block].ctypes.data + p_state[remote_block].nbytes
            assert any(
                base <= local_addr and local_addr + size <= base + backing.nbytes
                for base, backing in [(d_conv.ctypes.data, d_backing), (d_state.ctypes.data, d_state_backing)]
            )
            ctypes.memmove(local_addr, remote_addr, size)
        np.testing.assert_array_equal(p_backing, p_before)
        np.testing.assert_array_equal(p_state_backing, p_state_before)
    expected_backing = before_conv.copy()
    expected_view = np.lib.stride_tricks.as_strided(expected_backing, shape=d_conv.shape, strides=d_conv.strides)
    if layout == "SD":
        expected_view[local_block, :history] = expected_conv
        received = d_conv[local_block, :history]
    else:
        expected_view[local_block, :, :history] = expected_conv.T
        received = d_conv[local_block, :, :history].T
    np.testing.assert_array_equal(d_backing, expected_backing)
    expected_state_backing = before_state.copy()
    np.lib.stride_tricks.as_strided(expected_state_backing, shape=d_state.shape, strides=d_state.strides)[
        local_block
    ] = expected_state
    np.testing.assert_array_equal(d_state_backing, expected_state_backing)
    # First D verify window reads offset accepted-1=0, including D3's four
    # queries. Subsequent accepted windows read from newly written D scratch.
    query = np.arange((d_steps + 1) * d_width, dtype=np.float32).reshape(-1, d_width)
    actual_window = np.concatenate([received, query]).astype(np.float32)
    ref_window = np.concatenate([expected_conv, query]).astype(np.float32)
    weight = np.arange(1, 5, dtype=np.float32)[:, None]
    for i in range(d_steps + 1):
        np.testing.assert_array_equal(
            (actual_window[i : i + 4] * weight).sum(0), (ref_window[i : i + 4] * weight).sum(0)
        )
    # The update kernel stores old_history[1:] + the verified query window.
    # Next iteration selects accepted-1, so rejected draft rows are not used.
    scratch = np.concatenate([received[1:], query])
    for accepted in range(1, d_steps + 2):
        selected = scratch[accepted - 1 : accepted + 2]
        committed = np.concatenate([expected_conv, query[:accepted]])[-history:]
        np.testing.assert_array_equal(selected, committed)


@pytest.mark.parametrize("p_steps", [1, 3])
@pytest.mark.parametrize("mode", ["align", "none"])
def test_producer_selects_committed_state_with_its_own_draft_count(p_steps, mode):
    fn = _method(
        "mooncake_connector.py",
        "MooncakeConnectorScheduler",
        "_get_transfer_block_ids",
        cdiv=lambda a, b: (a + b - 1) // b,
    )
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode=mode),
            speculative_config=SimpleNamespace(num_speculative_tokens=p_steps),
        ),
        group_transfer_info=[SimpleNamespace(is_state_group=True, tokens_per_block=128)],
        dcp_size=1,
    )
    blocks = (
        [40, 41, 42] + [100 + i for i in range(p_steps)]
        if mode == "align"
        else [42] + [100 + i for i in range(p_steps)]
    )
    assert fn(scheduler, (blocks,), 300) == ([42],)


@pytest.mark.parametrize(
    "remote_rows,remote_width,remote_state_heads,remote_dtype",
    [(2, 12, 2, 2), (4, 6, 2, 2), (4, 12, 1, 2), (4, 12, 2, 4)],
)
def test_v1_rejects_incompatible_producer_before_transfer(remote_rows, remote_width, remote_state_heads, remote_dtype):
    fn = _method(
        "mooncake_connector.py",
        "KVCacheRecvingThread",
        "_append_mamba_transfer_meta",
        is_conv_state_dim_first=lambda: False,
    )
    config = SimpleNamespace(
        hf_text_config=SimpleNamespace(linear_attn_config={"num_heads": 16, "head_dim": 2, "short_conv_kernel_size": 4})
    )
    thread = SimpleNamespace(tp_size=8, vllm_config=SimpleNamespace(model_config=config))
    src, dst, sizes = [], [], []
    with pytest.raises(ValueError, match="Incompatible Mamba|KDA state shapes"):
        fn(
            thread,
            src,
            dst,
            sizes,
            group_spec={"shapes": [[6, 12], [2, 2, 2]], "dtype_sizes": [2, 4]},
            remote_group_spec={
                "shapes": [[remote_rows, remote_width], [remote_state_heads, 2, 2]],
                "dtype_sizes": [remote_dtype, 4],
            },
            src_layer_base_addr=[1000, 2000],
            dst_layer_base_addr=[3000, 4000],
            block_len=[144, 32],
            block_stride=[256, 256],
            remote_block_stride=[256, 256],
            remote_block_id=1,
            local_block_id=1,
            tp_num_need_pulls=1,
            remote_tp_offset=0,
        )
    assert (src, dst, sizes) == ([], [], [])


@pytest.mark.parametrize("local_len,remote_len", [(144, 48), (144, 192), (288, 96)])
def test_v2_rejects_truncated_or_different_dtype_conv_metadata(local_len, remote_len):
    fn = _method(
        "mooncake/pull_worker.py",
        "MooncakePullRecvingThread",
        "_append_mamba_transfer_addresses",
        torch=torch,
        MambaAttentionBackendEnum=_Backend,
    )
    config = SimpleNamespace(
        hf_text_config=SimpleNamespace(linear_attn_config={"num_heads": 16, "head_dim": 2, "short_conv_kernel_size": 4})
    )
    thread = SimpleNamespace(
        tp_size=8,
        tp_rank=0,
        model_config=config,
        layer_names=["kda"],
        kv_caches_base_addr=[[1000, 2000]],
        block_shapes=[[(6, 12), (2, 2, 2)]],
        block_lens=[[local_len, 32]],
        block_strides=[[512, 512]],
    )
    remote = SimpleNamespace(
        metadata_by_tp_rank={0: SimpleNamespace(kv_caches_base_addr=[[3000, 4000]])},
        block_shapes=[[(4, 12), (2, 2, 2)]],
        block_lens=[[remote_len, 32]],
        block_strides=[[512, 512]],
    )
    spec = SimpleNamespace(dtypes=(torch.float16, torch.float32), mamba_type=_Backend.GDN_ATTN)
    src, dst, sizes = [], [], []
    with pytest.raises(ValueError, match="byte lengths"):
        fn(thread, spec, 0, 8, {(0, 0): [("request", [1], [2])]}, remote, src, dst, sizes)
    assert (src, dst, sizes) == ([], [], [])
