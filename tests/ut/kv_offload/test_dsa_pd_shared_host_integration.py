# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import TestCase

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.worker import (
    MooncakeSFAPullRecvingThread,
)


class TestSharedHostIntegration(TestCase):
    def test_shared_host_ownership_is_applied_before_address_building(self):
        pool = SimpleNamespace(rank=3, size=8, contains=lambda ptr: ptr == 1024)
        local: list[int] = []
        remote: list[int] = []
        lengths: list[int] = []
        MooncakeSFAPullRecvingThread._append_block_transfer_addresses(
            [list(range(16))],
            [list(range(20, 36))],
            1024,
            100000,
            128,
            128,
            128,
            local,
            remote,
            lengths,
            host_kv=pool,
        )
        self.assertEqual(local, [1024 + 6 * 128])
        self.assertEqual(remote, [100000 + 26 * 128])
        self.assertEqual(lengths, [2 * 128])

    def test_strided_or_partial_blocks_keep_individual_byte_ranges(self):
        pool = SimpleNamespace(rank=1, size=3, contains=lambda ptr: ptr == 1024)
        for local_stride, remote_stride, length in ((256, 512, 64), (128, 256, 128), (256, 128, 128)):
            with self.subTest(local_stride=local_stride, remote_stride=remote_stride, length=length):
                local: list[int] = []
                remote: list[int] = []
                lengths: list[int] = []
                MooncakeSFAPullRecvingThread._append_block_transfer_addresses(
                    [list(range(8))],
                    [list(range(20, 28))],
                    1024,
                    100000,
                    local_stride,
                    remote_stride,
                    length,
                    local,
                    remote,
                    lengths,
                    local_inner_offset=16,
                    remote_inner_offset=32,
                    host_kv=pool,
                )
                self.assertEqual(local, [1024 + i * local_stride + 16 for i in (3, 4, 5)])
                self.assertEqual(remote, [100000 + i * remote_stride + 32 for i in (23, 24, 25)])
                self.assertEqual(lengths, [length] * 3)

    def test_rank_without_owned_host_blocks_emits_no_addresses(self):
        pool = SimpleNamespace(rank=7, size=8, contains=lambda ptr: ptr == 1024)
        local: list[int] = []
        remote: list[int] = []
        lengths: list[int] = []
        MooncakeSFAPullRecvingThread._append_block_transfer_addresses(
            [[1, 2]], [[10, 11]], 1024, 100000, 128, 128, 128, local, remote, lengths, host_kv=pool
        )
        self.assertEqual((local, remote, lengths), ([], [], []))

    def test_rank_local_hbm_destinations_keep_the_complete_transfer(self):
        pool = SimpleNamespace(rank=3, size=8, contains=lambda ptr: False)
        local: list[int] = []
        remote: list[int] = []
        lengths: list[int] = []
        MooncakeSFAPullRecvingThread._append_block_transfer_addresses(
            [list(range(16))],
            [list(range(20, 36))],
            1024,
            100000,
            128,
            128,
            128,
            local,
            remote,
            lengths,
            host_kv=pool,
        )
        self.assertEqual((local, remote, lengths), ([1024], [100000 + 20 * 128], [16 * 128]))
