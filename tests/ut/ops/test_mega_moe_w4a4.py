import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.mega_moe_w4a4 import (
    _BLOCK_DIM,
    _TILING_FIELDS,
    HADAMARD_BLOCK_SIZE,
    _make_tiling,
    pack_nz_int4,
    routing_prep,
)


class TestMegaMoeW4A4Host(TestBase):
    """CPU unit tests for the host-side orchestration (no NPU / kernel launch)."""

    def test_constants(self):
        self.assertEqual(HADAMARD_BLOCK_SIZE, 64)
        self.assertEqual(_BLOCK_DIM, 24)  # 910B (A2) cube core count

    def test_pack_nz_int4_shape_and_nibbles(self):
        # K % 16 == 0, N % 64 == 0 are the FRACTAL_NZ packing constraints.
        E, K, N = 2, 32, 128
        # Uniform value -> NZ reordering is irrelevant, so every packed byte is
        # the same nibble pair; lets us assert the pack math independent of layout.
        for val, expect in ((3, (3 & 0xF) | ((3 & 0xF) << 4)), (-1, -1)):
            w = torch.full((E, K, N), val, dtype=torch.int8)
            packed = pack_nz_int4(w)
            self.assertEqual(packed.dtype, torch.int8)
            self.assertEqual(packed.numel(), E * K * N // 2)  # 2 int4 per byte
            self.assertTrue(bool((packed == expect).all()))

    def test_make_tiling_fields(self):
        t = _make_tiling(128, 2048, 256, 256, base_k=256, base_m=128)
        self.assertEqual(len(t), 50)  # 50 int32 TCubeTiling fields
        d = dict(zip(_TILING_FIELDS, t.tolist()))
        self.assertEqual(d["usedCoreNum"], _BLOCK_DIM)
        self.assertEqual((d["M"], d["N"], d["Ka"], d["Kb"]), (128, 256, 2048, 2048))
        # baseM MUST equal singleCoreM for the int4 MatmulImpl (else it corrupts).
        self.assertEqual(d["baseM"], d["singleCoreM"])
        self.assertEqual(d["baseM"], 128)

    def test_routing_prep_correctness(self):
        # flat top-k ids = [0,2, 1,2, 0,1]; each expert appears twice.
        topk_ids = torch.tensor([[0, 2], [1, 2], [0, 1]], dtype=torch.int32)
        num_experts = 3
        group_list, eri, sort_idx = routing_prep(topk_ids, num_experts)

        # group_list = cumulative per-expert counts.
        self.assertEqual(group_list.tolist(), [2, 4, 6])
        self.assertEqual(group_list.dtype, torch.int64)

        m = topk_ids.numel()
        # sort_idx is a permutation of [0, M).
        self.assertEqual(sorted(sort_idx.tolist()), list(range(m)))
        # eri is the inverse permutation of sort_idx.
        inv = torch.empty(m, dtype=torch.long)
        inv[sort_idx.long()] = torch.arange(m)
        self.assertEqual(eri.long().tolist(), inv.tolist())
        self.assertEqual(eri.dtype, torch.int32)
        self.assertEqual(sort_idx.dtype, torch.int32)
