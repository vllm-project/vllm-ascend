"""CPU wire-contract coverage for producer Off -> decoder RecoverSSM."""

import ctypes
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker as module
from vllm_ascend.models.qwen4_exp.cache_config import make_slab_view


def spec(recover=False):
    shapes = ((6, 5120), (24, 128, 128))
    dtypes = (torch.bfloat16, torch.bfloat16)
    if recover:
        shapes += ((24, 4, 257),)
        dtypes += (torch.float32,)
    return MambaSpec(
        block_size=1024,
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        num_speculative_blocks=0 if recover else 3,
    )


def worker(monkeypatch, recover):
    descriptor = spec(recover)
    backing = torch.zeros(3 * 1048576, dtype=torch.int8)
    views = []
    offset = 0
    for shape, dtype in zip(descriptor.shapes, descriptor.dtypes):
        views.append(
            make_slab_view(
                backing, dtype=dtype, num_blocks=3, item_shape=shape, storage_offset=offset, block_stride=1048576
            )
        )
        offset += views[-1][0].numel() * views[-1].element_size()
    w = module.MooncakeBaseConnectorWorker.__new__(module.MooncakeBaseConnectorWorker)
    w.__dict__.update(
        kv_cache_config=NS(num_blocks=3, kv_cache_tensors=[NS(size=backing.numel(), layers=["gdn"])]),
        vllm_config=NS(parallel_config=NS(cp_kv_cache_interleave_size=1)),
        ascend_config=NS(kvpp_config=NS(size=1)),
        tp_rank=0,
        tp_size=2,
        pcp_rank=0,
        pcp_size=1,
        layer_name_to_group_index={"gdn": 0},
        layer_name_to_spec_index={"gdn": 0},
        kv_cache_specs=[descriptor],
        engine_id="on" if recover else "off",
        te_rpc_port=1,
        block_size=1024,
        side_channel_host="127.0.0.1",
        handshake_port=2,
    )
    w._build_kv_cache_spec_mappings = MagicMock()
    monkeypatch.setattr(module.global_te, "register_buffer", MagicMock())
    w.register_kv_caches({"gdn": views})
    return w, views, descriptor


def wire_worker(w):
    m = w.transfer_metadata
    return NS(
        tp_rank=0,
        tp_size=2,
        pcp_rank=0,
        pcp_size=1,
        layer_names=m.layer_names,
        kv_caches_base_addr=m.kv_caches_base_addr,
        block_shapes=m.block_shapes,
        block_strides=m.block_strides,
        block_lens=m.block_lens,
    )


def remote(w):
    m = w.transfer_metadata
    return NS(
        metadata_by_pcp_rank={0: NS(metadata_by_tp_rank={0: m})},
        block_shapes=m.block_shapes,
        block_strides=m.block_strides,
        block_lens=m.block_lens,
    )


@pytest.mark.parametrize(
    "sender_recover,receiver_recover", [(False, False), (False, True), (True, False), (True, True)]
)
def test_off_and_recovery_share_two_state_wire_and_replay_stays_local(monkeypatch, sender_recover, receiver_recover):
    a, av, a_spec = worker(monkeypatch, sender_recover)
    b, bv, b_spec = worker(monkeypatch, receiver_recover)
    assert len(a.transfer_metadata.kv_caches_base_addr[0]) == 2
    assert len(b.transfer_metadata.kv_caches_base_addr[0]) == 2
    assert len(av) == len(a_spec.shapes) == 2 + int(sender_recover)
    assert len(bv) == len(b_spec.shapes) == 2 + int(receiver_recover)
    for i, value in enumerate((3, 7)):
        av[i][1].fill_(value)
    if receiver_recover:
        bv[2].fill_(11)
    src, dst, sizes = [], [], []
    module.MooncakeTransferAddressMixin._append_mamba_transfer_addresses(
        wire_worker(a), a_spec, 0, 0, 2, {(0, 0): [("push", [1], [2])]}, remote(b), src, dst, sizes
    )
    assert sizes == [6 * 5120 * 2, 24 * 128 * 128 * 2]
    assert src == [view.data_ptr() + 1048576 for view in av[:2]]
    assert dst == [view.data_ptr() + 2 * 1048576 for view in bv[:2]]
    for source, target, size in zip(src, dst, sizes):
        ctypes.memmove(target, source, size)
    for i, value in enumerate((3, 7)):
        assert torch.all(bv[i][2] == value)
        assert torch.count_nonzero(bv[i][0]) == 0
    if receiver_recover:
        assert torch.all(bv[2] == 11)
    assert len(a.kv_caches["gdn"]) == 2 + int(sender_recover)
    assert len(b.kv_caches["gdn"]) == 2 + int(receiver_recover)


def test_projection_preserves_generic_mamba_and_original_descriptor():
    baseline = spec(False)
    assert module._persistent_mamba_transfer_spec(baseline) is baseline
    recover = spec(True)
    wire = module._persistent_mamba_transfer_spec(recover)
    assert len(wire.shapes) == len(wire.dtypes) == 2
    assert len(recover.shapes) == len(recover.dtypes) == 3
    ple = MambaSpec(block_size=1024, shapes=((16, 32),), dtypes=(torch.bfloat16,), tp_replicated=True)
    assert module._persistent_mamba_transfer_spec(ple) is ple
    generic = MambaSpec(block_size=1024, shapes=recover.shapes, dtypes=recover.dtypes)
    assert module._persistent_mamba_transfer_spec(generic) is generic


def test_unrecognized_gdn_replay_dtype_is_rejected():
    from dataclasses import replace

    bad = replace(spec(True), dtypes=(torch.bfloat16, torch.bfloat16, torch.bfloat16))
    with pytest.raises(ValueError, match="FP32 third state"):
        module._persistent_mamba_transfer_spec(bad)
