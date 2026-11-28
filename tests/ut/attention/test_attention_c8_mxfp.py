from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackendImpl
from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata


class TestC8MXFPQfaQueryPlan(TestBase):
    """What the two QFA operators are planned against for one step.

    max_seqlen_q is the longest single query, not the batch total: the
    metadata op seeds querySeqSize with it and then raises it with
    max(attr, per-request length), so an inflated attr is never walked back.
    The q scale layout then picks the kernel -- TND compiles the prefill
    template, N2TGD the decode one -- and both operators must be given the
    same one.
    """

    NUM_KV_HEADS = 2
    GROUP_SIZE = 4
    NUM_HEADS = NUM_KV_HEADS * GROUP_SIZE
    D_GROUPS = 2  # head_dim // 64

    def setUp(self):
        self.impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        self.impl.sliding_window = None
        self.impl.num_heads = self.NUM_HEADS
        self.impl.num_kv_heads = self.NUM_KV_HEADS

    def _forward(self, query_start_loc, max_query_len):
        """Run one step and report what each operator was handed."""
        num_tokens = int(query_start_loc[-1])
        # Distinct value per (token, head, group, element) so the N2TGD
        # permutation can be checked entry by entry.
        scale = (
            torch.arange(num_tokens * self.NUM_HEADS * self.D_GROUPS * 2, dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .view(num_tokens, self.NUM_HEADS, self.D_GROUPS, 2)
        )
        attn_metadata = SimpleNamespace(
            causal=True,
            block_tables=None,
            query_start_loc_gpu=torch.tensor(query_start_loc, dtype=torch.int32),
            seq_lens_gpu=torch.ones(len(query_start_loc) - 1, dtype=torch.int32),
            max_query_len=max_query_len,
        )
        seen = {"source_scale": scale}

        def fake_metadata(
            _self, _metadata, *, cu_seqlens_q, seqused_kv, max_seqlen_q, mask_mode, layout_q_descale, partition
        ):
            seen["metadata_max_seqlen_q"] = max_seqlen_q
            seen["metadata_mask_mode"] = mask_mode
            seen["metadata_layout"] = layout_q_descale
            return MagicMock()

        def fake_run(
            _self,
            _query,
            query_scale,
            _kv_cache,
            _attn_metadata,
            *,
            max_seqlen_q,
            mask_mode,
            layout_q_descale,
            output,
            **kwargs,
        ):
            seen["max_seqlen_q"] = max_seqlen_q
            seen["mask_mode"] = mask_mode
            seen["layout"] = layout_q_descale
            seen["scale"] = query_scale
            return output

        with (
            patch.object(AscendC8MXFPAttentionBackendImpl, "_get_qfa_metadata", fake_metadata),
            patch.object(AscendC8MXFPAttentionBackendImpl, "_run_qfa", fake_run),
        ):
            self.impl._forward_mxfp8_attention(
                torch.zeros(num_tokens, self.NUM_HEADS, 8, dtype=torch.uint8),
                scale,
                (MagicMock(), MagicMock(), MagicMock(), MagicMock()),
                attn_metadata,
                torch.zeros(num_tokens, self.NUM_HEADS, 8),
            )

        # A plan computed for one kernel must not be fed to the other.
        self.assertEqual(seen["metadata_max_seqlen_q"], seen["max_seqlen_q"])
        self.assertEqual(seen["metadata_layout"], seen["layout"])
        self.assertEqual(seen["metadata_mask_mode"], seen["mask_mode"])
        return seen

    # --- max_seqlen_q -----------------------------------------------------

    def test_uniform_decode_does_not_declare_the_batch_total(self):
        # 4 requests, one query token each: the old code declared 4.
        self.assertEqual(self._forward([0, 1, 2, 3, 4], max_query_len=1)["max_seqlen_q"], 1)

    def test_mtp_decode_uses_the_draft_query_length(self):
        # 3 requests x 2 tokens (1 + 1 draft): the old code declared 6.
        self.assertEqual(self._forward([0, 2, 4, 6], max_query_len=2)["max_seqlen_q"], 2)

    def test_mixed_batch_uses_the_longest_prefill(self):
        # One 5-token prefill plus 3 decodes; the longest query is the prefill.
        self.assertEqual(self._forward([0, 5, 6, 7, 8], max_query_len=5)["max_seqlen_q"], 5)

    def test_missing_max_query_len_falls_back_to_the_token_count(self):
        self.assertEqual(self._forward([0, 1, 2, 3, 4], max_query_len=None)["max_seqlen_q"], 4)

    # --- q scale layout ---------------------------------------------------

    def test_decode_sends_the_scale_as_n2tgd(self):
        seen = self._forward([0, 1, 2, 3, 4], max_query_len=1)
        self.assertEqual(seen["layout"], "N2TGD")
        num_tokens = seen["source_scale"].shape[0]
        self.assertEqual(
            tuple(seen["scale"].shape),
            (self.NUM_KV_HEADS, num_tokens, self.GROUP_SIZE, self.D_GROUPS, 2),
        )

    def test_n2tgd_keeps_every_head_scale_with_its_kv_head(self):
        seen = self._forward([0, 1, 2, 3, 4], max_query_len=1)
        source, permuted = seen["source_scale"], seen["scale"]
        for token in range(source.shape[0]):
            for head in range(self.NUM_HEADS):
                # Query heads are GQA-contiguous: head n serves kv head n // G.
                kv_head, group = divmod(head, self.GROUP_SIZE)
                self.assertTrue(
                    torch.equal(permuted[kv_head, token, group], source[token, head]),
                    f"token={token} head={head} lost its scale",
                )

    def test_prefill_keeps_the_scale_in_tnd_untouched(self):
        # G=4, so a 21-token query crosses the operator's G*Q_S boundary.
        seen = self._forward([0, 21], max_query_len=21)
        self.assertEqual(seen["layout"], "TND")
        self.assertIs(seen["scale"], seen["source_scale"])

    def test_boundary_value_still_takes_the_decode_layout(self):
        # G*Q_S == 80 exactly; the operator doc recommends N2TGD at or below.
        self.assertEqual(self._forward([0, 20], max_query_len=20)["layout"], "N2TGD")

    def test_mtp_verify_queries_take_the_decode_layout(self):
        # MTP verify steps are SpecDecoding with 1+spec query tokens each;
        # the layout decision reads the query shape, not the scheduler
        # state, so the decode kernel applies here too. Regression guard
        # against keying this on attn_state == DecodeOnly, which never
        # fires once MTP is enabled.
        seen = self._forward([0, 4, 8], max_query_len=4)
        self.assertEqual(seen["layout"], "N2TGD")
        self.assertEqual(seen["max_seqlen_q"], 4)

    def test_non_divisible_heads_stay_in_tnd(self):
        # MTP draft layers can run a head count that does not split into
        # whole kv-head groups; the group reshape would miscount elements
        # and raise, so the guard falls back to TND instead.
        self.impl.num_kv_heads = 3  # 8 % 3 != 0
        seen = self._forward([0, 1, 2, 3, 4], max_query_len=1)
        self.assertEqual(seen["layout"], "TND")
        self.assertIs(seen["scale"], seen["source_scale"])

    def test_zero_kv_heads_stay_in_tnd(self):
        # Degenerate layers must not hit a ZeroDivisionError in the guard.
        self.impl.num_kv_heads = 0
        seen = self._forward([0, 1, 2, 3, 4], max_query_len=1)
        self.assertEqual(seen["layout"], "TND")
        self.assertIs(seen["scale"], seen["source_scale"])

    # --- mask mode --------------------------------------------------------

    def test_uniform_decode_drops_the_mask(self):
        # One query row per request: CAUSAL constrains nothing there (the KV
        # range is bounded by seqused_kv), and NO_MASK is a separate tiling
        # key whose kernel never reads the mask.
        self.assertEqual(self._forward([0, 1, 2, 3, 4], max_query_len=1)["mask_mode"], 0)

    def test_mtp_verify_keeps_the_causal_mask(self):
        # 1 + num_spec query rows per request: row 0 must not see row 1's KV.
        self.assertEqual(self._forward([0, 2, 4, 6], max_query_len=2)["mask_mode"], 3)

    def test_prefill_keeps_the_causal_mask(self):
        self.assertEqual(self._forward([0, 5, 6, 7, 8], max_query_len=5)["mask_mode"], 3)

    def test_unknown_query_length_keeps_the_causal_mask(self):
        # max_query_len absent falls back to the token total, which is only
        # 1 for a genuinely single-token step -- never a silent NO_MASK.
        self.assertEqual(self._forward([0, 1, 2, 3, 4], max_query_len=None)["mask_mode"], 3)


class TestC8MXFPPerStepDerivations(TestBase):
    """Per-step quantities are derived once, not once per layer.

    cu_seqlens_q / seqused_kv and the K-scale slot decomposition depend only
    on the runner's per-step buffers, but a step walks 23 full-attention
    layers. Caching them on the shared AscendMetadata is what keeps that from
    being ~200 redundant device ops a decode step, so the reuse is the thing
    worth pinning.
    """

    def setUp(self):
        self.impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        self.attn_metadata = SimpleNamespace(
            query_start_loc_gpu=torch.tensor([0, 4, 8], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([10, 20], dtype=torch.int32),
        )

    def test_lengths_are_derived_once_per_step(self):
        first = self.impl._qfa_step_lengths(self.attn_metadata, 8)
        second = self.impl._qfa_step_lengths(self.attn_metadata, 8)
        # Same tensor objects, so later layers add no ops to the graph.
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_lengths_are_sanitized(self):
        # -1 padding clamps to 0, and cummax keeps the boundaries monotonic
        # so cu_seqlens_q[-1] still equals the token total.
        self.attn_metadata.query_start_loc_gpu = torch.tensor([0, 4, 8, -1, -1], dtype=torch.int32)
        self.attn_metadata.seq_lens_gpu = torch.tensor([10, 20, 0, 0], dtype=torch.int32)
        cu_seqlens_q, seqused_kv = self.impl._qfa_step_lengths(self.attn_metadata, 8)
        self.assertEqual(cu_seqlens_q.tolist(), [0, 4, 8, 8, 8])
        self.assertEqual(seqused_kv.tolist(), [10, 20, 1, 1])

    def test_k_scale_slot_index_is_derived_once_per_step(self):
        slots = torch.tensor([2, 5, -1], dtype=torch.int64)
        first = self.impl._qfa_k_scale_slot_index(self.attn_metadata, slots, 4)
        second = self.impl._qfa_k_scale_slot_index(self.attn_metadata, slots, 4)
        for a, b in zip(first, second):
            self.assertIs(a, b)

    def test_k_scale_slot_index_decomposes_slots(self):
        # block_size 4, K-scale token fragment 16: slot 5 -> block 1,
        # offset 1 -> segment 0, fragment 1. The padded row clamps to slot 0,
        # the null block's, which keeps every shape static without a mask.
        slots = torch.tensor([2, 5, -1], dtype=torch.int64)
        block_ids, seg_ids, frag_ids = self.impl._qfa_k_scale_slot_index(self.attn_metadata, slots, 4)
        self.assertEqual(block_ids.tolist(), [0, 1, 0])
        self.assertEqual(seg_ids.tolist(), [0, 0, 0])
        self.assertEqual(frag_ids.tolist(), [2, 1, 0])

    def test_the_two_derivations_do_not_collide_in_the_step_cache(self):
        lengths = self.impl._qfa_step_lengths(self.attn_metadata, 8)
        slots = self.impl._qfa_k_scale_slot_index(self.attn_metadata, torch.tensor([2, 5, -1], dtype=torch.int64), 4)
        self.assertIs(self.impl._qfa_step_lengths(self.attn_metadata, 8)[0], lengths[0])
        self.assertIs(
            self.impl._qfa_k_scale_slot_index(self.attn_metadata, torch.tensor([2, 5, -1], dtype=torch.int64), 4)[0],
            slots[0],
        )


class TestC8MXFPQfaMetadataPlan(TestBase):
    """One AICPU metadata plan a step, capture included.

    The plan's inputs are head counts, head dim, quant mode, the two length
    tensors, mask mode, window and layouts -- nothing layer-specific -- and
    the main operator declares ``metadata`` a read-only Input. So the step's
    full-attention layers share one plan instead of each paying an
    AICore-to-AICPU round trip. That holds while a graph is captured too: the
    deriving call is recorded inside the captured region, ahead of every read
    of its output.
    """

    LAYERS_PER_STEP = 23

    def setUp(self):
        self.impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        self.impl.num_heads = 8
        self.impl.num_kv_heads = 2
        self.impl.head_size = 128

    def _plans(self, calls):
        """Issue one step's worth of plans; return (plans, operator calls)."""
        attn_metadata = SimpleNamespace()
        invocations = []

        def metadata_op(*args, **kwargs):
            invocations.append((args, kwargs))
            return MagicMock()

        defaults = {
            "cu_seqlens_q": torch.tensor([0, 1, 2], dtype=torch.int32),
            "seqused_kv": torch.tensor([10, 20], dtype=torch.int32),
            "max_seqlen_q": 1,
            "mask_mode": 0,
            "layout_q_descale": "N2TGD",
        }
        ops_module = MagicMock(quant_flash_attn_metadata=metadata_op)
        with patch.dict("sys.modules", {"cann_ops_transformer.ops": ops_module}):
            plans = [self.impl._get_qfa_metadata(attn_metadata, **{**defaults, **call}) for call in calls]
        return plans, invocations

    def test_one_plan_serves_every_layer(self):
        plans, invocations = self._plans([{}] * self.LAYERS_PER_STEP)
        self.assertEqual(len(invocations), 1)
        for plan in plans[1:]:
            self.assertIs(plan, plans[0])

    def test_capture_shares_the_plan_too(self):
        # Regression guard for the code this replaced: the old design
        # bypassed the cache whenever a capture was in flight, which put one
        # AICPU op per layer into the captured graph and replayed all of
        # them every step. The shared derivation now runs inside the
        # captured region by construction -- there is no bypass path left.
        plans, invocations = self._plans([{}] * self.LAYERS_PER_STEP)
        self.assertEqual(len(invocations), 1)
        for plan in plans[1:]:
            self.assertIs(plan, plans[0])

    def test_a_different_non_tensor_input_gets_its_own_plan(self):
        for override in ({"layout_q_descale": "TND"}, {"mask_mode": 3}, {"max_seqlen_q": 2}):
            with self.subTest(**override):
                plans, invocations = self._plans([{}, override])
                self.assertEqual(len(invocations), 2)
                self.assertIsNot(plans[0], plans[1])

    def test_the_step_mask_mode_reaches_the_operator(self):
        # The plan is a schedule for one kernel variant, so the mask mode it
        # was built for has to be the one the main call then uses.
        _, invocations = self._plans([{"mask_mode": 3}])
        self.assertEqual(invocations[0][1]["mask_mode"], 3)


class TestMixedQFASplit(TestBase):
    def setUp(self):
        self.impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        self.impl.num_heads = 8
        self.impl.num_kv_heads = 2
        self.impl.head_size = 128
        self.impl.scale = 128**-0.5
        self.impl.sliding_window = None
        self.impl.vllm_config = SimpleNamespace(
            kv_transfer_config=None,
            model_config=SimpleNamespace(enforce_eager=True),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
            speculative_config=None,
        )
        self.runtime_mode = CUDAGraphMode.NONE
        self.query = torch.arange(24).view(24, 1, 1).expand(24, 8, 128).to(torch.bfloat16).contiguous()
        self.scale = torch.arange(24 * 8 * 4, dtype=torch.int32).to(torch.uint8).view(24, 8, 2, 2)
        self.kv = (torch.empty(1), torch.empty(1), torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8))
        self.metadata = AscendMetadata(
            num_decodes=2,
            num_decode_tokens=2,
            num_prefills=1,
            num_actual_tokens=24,
            attn_state=AscendAttentionState.ChunkedPrefill,
            max_query_len=22,
            actual_seq_lengths_q=[1, 2, 24],
            query_start_loc_gpu=torch.tensor([0, 1, 2, 24], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([141, 79, 67], dtype=torch.int32),
            block_tables=torch.tensor([[8], [1], [4]], dtype=torch.int32),
            attn_mask=torch.ones(2, 2, dtype=torch.int8),
        )
        self.ops = MagicMock()
        self.ops.quant_flash_attn_metadata.side_effect = lambda *args, **kwargs: object()
        self.ops.quant_flash_attn.side_effect = self._fake_qfa

    def _fake_qfa(self, query, key, value, query_scale, *_args, **kwargs):
        # Verify which KV rows, Q scale rows and query offsets belong to
        # each token. Reconstruct the TND scale before comparing outputs.
        self.assertIs(key, self.kv[0])
        self.assertIs(value, self.kv[1])
        scale = query_scale.view(torch.uint8)
        if kwargs["layout_q_descale"] == "N2TGD":
            scale = scale.permute(1, 0, 2, 3, 4).reshape_as(self.scale[: len(query)])
        lens = kwargs["cu_seqlens_q"].diff().tolist()
        offsets = kwargs["cu_seqlens_q"].tolist()
        self.assertEqual((offsets[0], offsets[-1]), (0, len(query)))
        result = query.float() + scale[:, :, 0, :1].float()
        for index in range(len(lens)):
            result[offsets[index] : offsets[index + 1]] += kwargs["seqused_kv"][index] + kwargs["block_table"][index, 0]
        return result.to(torch.bfloat16), None

    def _forward(self):
        output = torch.full((26, 8, 128), -1, dtype=torch.bfloat16)
        with (
            patch("vllm_ascend.attention.attention_c8_mxfp.record_attention_compute_start"),
            patch(
                "vllm_ascend.attention.attention_c8_mxfp.get_forward_context",
                return_value=SimpleNamespace(cudagraph_runtime_mode=self.runtime_mode),
            ),
            patch(
                "vllm_ascend.attention.attention_c8_mxfp.torch_npu.npu_dynamic_mx_quant",
                return_value=(self.query, self.scale),
            ),
            patch.dict("sys.modules", {"cann_ops_transformer.ops": self.ops}),
        ):
            self.assertIs(self.impl.forward(None, self.query, None, None, self.kv, self.metadata, output), output)
        self.assertTrue((output[24:] == -1).all())
        return output

    def test_eager_mixed_output_and_plan_reuse(self):
        self.impl.vllm_config.model_config.enforce_eager = False
        baseline = self._forward()
        self.impl.vllm_config.model_config.enforce_eager = True
        self.ops.reset_mock()
        for _ in range(2):  # Two layers sharing this step's metadata.
            torch.testing.assert_close(self._forward(), baseline, rtol=0, atol=0)
        self.assertEqual(self.ops.quant_flash_attn_metadata.call_count, 2)
        decode, prefill, next_decode, next_prefill = self.ops.quant_flash_attn.call_args_list
        self.assertEqual(decode.kwargs["cu_seqlens_q"].tolist(), [0, 1, 2])
        self.assertEqual(prefill.kwargs["cu_seqlens_q"].tolist(), [0, 22])
        self.assertEqual((decode.kwargs["mask_mode"], decode.kwargs["layout_q_descale"]), (3, "N2TGD"))
        self.assertEqual(decode.args[3].shape, (2, 2, 4, 2, 2))
        decode_plan, prefill_plan = self.ops.quant_flash_attn_metadata.call_args_list
        self.assertEqual(decode_plan.kwargs["layout_q_descale"], "N2TGD")
        self.assertEqual(prefill_plan.kwargs["layout_q_descale"], "TND")
        self.assertEqual((prefill.kwargs["mask_mode"], prefill.kwargs["layout_q_descale"]), (3, "TND"))
        self.assertIs(decode.kwargs["attn_mask"], self.metadata.attn_mask)
        self.assertEqual(decode.kwargs["max_seqlen_q"], 1)
        self.assertEqual(prefill.kwargs["max_seqlen_q"], 22)
        self.assertIs(prefill.kwargs["attn_mask"], self.metadata.attn_mask)
        self.assertIs(decode.kwargs["metadata"], next_decode.kwargs["metadata"])
        self.assertIs(prefill.kwargs["metadata"], next_prefill.kwargs["metadata"])
        self.assertIsNot(decode.kwargs["metadata"], prefill.kwargs["metadata"])

    def test_metadata_lengths_have_partition_sized_storage(self):
        self.metadata = replace(
            self.metadata,
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=2,
            max_query_len=15,
            actual_seq_lengths_q=[1, 9, 24],
            query_start_loc_gpu=torch.tensor([0, 1, 9, 24], dtype=torch.int32),
        )
        self._forward()
        plans = self.ops.quant_flash_attn_metadata.call_args_list
        self.assertEqual(len(plans), 2)
        for plan, offsets, lengths in zip(plans, ([0, 1], [0, 8, 23]), ([141], [79, 67])):
            for name, expected in (("cu_seqlens_q", offsets), ("seqused_kv", lengths)):
                with self.subTest(name=name, expected=expected):
                    tensor = plan.kwargs[name]
                    self.assertEqual(tensor.tolist(), expected)
                    self.assertEqual(tensor.storage_offset(), 0)
                    self.assertEqual(tensor.untyped_storage().nbytes(), len(expected) * tensor.element_size())

    def test_full_decode_only_distinguishes_runtime_modes(self):
        self.impl.vllm_config.model_config.enforce_eager = False
        self.impl.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        for mode, expected_calls in ((CUDAGraphMode.FULL, 1), (CUDAGraphMode.PIECEWISE, 1), (CUDAGraphMode.NONE, 2)):
            with self.subTest(mode=mode):
                self.runtime_mode = mode
                self.metadata = replace(self.metadata, qfa_metadata_cache={})
                self.ops.reset_mock()
                self._forward()
                self.assertEqual(self.ops.quant_flash_attn.call_count, expected_calls)

    def test_full_decode_only_keeps_pure_decode_whole(self):
        self.impl.vllm_config.model_config.enforce_eager = False
        self.impl.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        self.runtime_mode = CUDAGraphMode.FULL
        num_reqs = len(self.query)
        self.metadata = replace(
            self.metadata,
            attn_state=AscendAttentionState.DecodeOnly,
            num_decodes=num_reqs,
            num_decode_tokens=num_reqs,
            num_prefills=0,
            max_query_len=1,
            actual_seq_lengths_q=list(range(1, num_reqs + 1)),
            query_start_loc_gpu=torch.arange(num_reqs + 1, dtype=torch.int32),
            seq_lens_gpu=torch.full((num_reqs,), 32, dtype=torch.int32),
            block_tables=torch.arange(num_reqs, dtype=torch.int32).reshape(-1, 1),
        )
        self._forward()
        self.assertEqual(self.ops.quant_flash_attn.call_count, 1)
        self.assertEqual(self.ops.quant_flash_attn.call_args.kwargs["mask_mode"], 0)

    def test_ineligible_steps_keep_one_call(self):
        base_metadata = self.metadata
        for overrides, eager, speculative in (
            ({}, False, None),
            ({}, True, object()),
            ({"num_prefills": 0}, True, None),
            ({"num_decodes": 0}, True, None),
            ({"num_decode_tokens": 3}, True, None),
            ({"actual_seq_lengths_q": [1, 2, 24, 24]}, True, None),
            ({"actual_seq_lengths_q": [1, 23, 24]}, True, None),
        ):
            with self.subTest(overrides=overrides, eager=eager, speculative=speculative):
                self.metadata = replace(self.metadata, **overrides, qfa_metadata_cache={})
                self.impl.vllm_config.model_config.enforce_eager = eager
                self.impl.vllm_config.speculative_config = speculative
                self._forward()
                self.assertEqual(self.ops.quant_flash_attn.call_count, 1)
                self.ops.reset_mock()
                self.metadata = base_metadata
