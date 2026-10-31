# SPDX-License-Identifier: Apache-2.0
"""SD byte-copy oracle for the existing Mooncake Mamba transfer path."""

import struct
from dataclasses import replace
from math import prod

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pull_worker

from .helpers import make_mamba_spec, make_pp_metadata
from .test_pull_worker import make_thread


def state_payload(heads, rank):
    # Encode global coordinates independently of the connector's offsets.
    conv = [
        row * 96 + projection * 32 + head * 4 + channel
        for row in range(8)
        for projection in range(3)
        for head in range(rank * heads, (rank + 1) * heads)
        for channel in range(4)
    ]
    state = [head * 16 + element for head in range(rank * heads, (rank + 1) * heads) for element in range(16)]
    return [struct.pack(f"<{len(conv)}H", *conv), struct.pack(f"<{len(state)}I", *state)]


def transfer_case(local_tp, remote_tp, remote_pcp, rank, remote_recover):
    def geometry(shards, recover, padding):
        heads = 8 // shards
        shapes = [(8, 12 * heads), (heads, 4, 4)]
        sizes = [2, 4]
        if recover:
            shapes += [(heads, 6, 4), (heads, 6, 8)]
            sizes += [4, 2]
        lengths = [prod(shape) * size for shape, size in zip(shapes, sizes)]
        return shapes, lengths, [length + padding for length in lengths]

    shapes, lengths, strides = geometry(local_tp, True, 64)
    remote_shapes, remote_lengths, remote_strides = geometry(remote_tp * remote_pcp, remote_recover, 128)
    local_bases = [10000 * (index + 1) for index in range(4)]
    remote_bases = [100000 * (index + 1) for index in range(len(remote_shapes))]
    spec = replace(
        make_mamba_spec(),
        shapes=tuple(shapes),
        dtypes=(torch.bfloat16, torch.float32, torch.float32, torch.bfloat16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    thread = make_thread(
        tp_size=local_tp,
        tp_rank=rank,
        recover_kda=True,
        kv_caches_base_addr=[local_bases],
        block_shapes=[shapes],
        block_strides=[strides],
        block_lens=[lengths],
    )
    remote = make_pp_metadata(
        block_shapes=[remote_shapes],
        block_lens=[remote_lengths],
        block_strides=[remote_strides],
        pcp_tp_base_addrs={pcp: {tp: [remote_bases] for tp in range(remote_tp)} for pcp in range(remote_pcp)},
    )
    return thread, spec, remote, local_bases, remote_bases


@pytest.mark.parametrize(
    ("local_tp", "remote_tp", "remote_pcp"), [(1, 1, 1), (2, 2, 1), (1, 4, 1), (4, 1, 1), (2, 1, 4), (8, 1, 2)]
)
@pytest.mark.parametrize("remote_recover", [False, True])
def test_recoverssm_sd_copy_reconstructs_state_and_preserves_scratch(
    monkeypatch, local_tp, remote_tp, remote_pcp, remote_recover
):
    monkeypatch.setattr(pull_worker, "is_conv_state_dim_first", lambda: False)
    for rank in range(local_tp):
        thread, spec, remote, local_bases, remote_bases = transfer_case(
            local_tp, remote_tp, remote_pcp, rank, remote_recover
        )
        # Dirty destination, including other pages, padding and verify records.
        destination = [bytearray([0xA5] * (stride * 3)) for stride in thread.block_strides[0]]
        expected = [buffer.copy() for buffer in destination]
        for index, payload in enumerate(state_payload(8 // local_tp, rank)):
            offset = thread.block_strides[0][index]
            expected[index][offset : offset + len(payload)] = payload
        for (worker_rank,) in thread._get_mamba_remote_tp_rank_groups(remote_pcp, remote_tp):
            tp_rank, pcp_rank = divmod(worker_rank, remote_pcp)
            source = [bytearray([0xCC] * (stride * 3)) for stride in remote.block_strides[0]]
            for index, payload in enumerate(state_payload(8 // (remote_tp * remote_pcp), worker_rank)):
                offset = 2 * remote.block_strides[0][index]
                source[index][offset : offset + len(payload)] = payload
            local_addrs, remote_addrs, sizes = [], [], []
            thread._append_mamba_transfer_addresses(
                spec,
                pcp_rank,
                tp_rank,
                remote_tp,
                {(0, 0): [("request", [1], [2])]},
                remote,
                local_addrs,
                remote_addrs,
                sizes,
            )
            assert len(sizes) == (2 if local_tp == remote_tp * remote_pcp else 25)
            for local_addr, remote_addr, size in zip(local_addrs, remote_addrs, sizes):
                index = next(i for i, base in enumerate(local_bases) if base <= local_addr < base + len(destination[i]))
                assert index < 2
                local_offset, remote_offset = local_addr - local_bases[index], remote_addr - remote_bases[index]
                assert 0 <= remote_offset < remote_offset + size <= len(source[index])
                destination[index][local_offset : local_offset + size] = source[index][
                    remote_offset : remote_offset + size
                ]
        assert destination == expected


@pytest.mark.parametrize("invalid", ["disabled", "count", "shape", "history", "block", "ds", "state"])
def test_recoverssm_transfer_rejects_incompatible_base_states(monkeypatch, invalid):
    monkeypatch.setattr(pull_worker, "is_conv_state_dim_first", lambda: invalid == "ds")
    thread, spec, remote, _, _ = transfer_case(1, 1, 1, 0, False)
    if invalid == "disabled":
        thread.recover_kda = False
    elif invalid == "count":
        remote.block_shapes[0].append((1,))
    elif invalid == "shape":
        remote.block_shapes[0][0] = (4, 192)  # Equal bytes, different SD shape.
    elif invalid == "history":
        remote.block_shapes[0][0] = (7, 96)
    elif invalid == "block":
        remote.layer_block_sizes[0] *= 2
    elif invalid == "state":
        remote.block_shapes[0][1] = (8, 2, 8)  # Equal bytes, different state axes.
    with pytest.raises(ValueError):
        thread._append_mamba_transfer_addresses(spec, 0, 0, 1, {(0, 0): [("request", [1], [2])]}, remote, [], [], [])
