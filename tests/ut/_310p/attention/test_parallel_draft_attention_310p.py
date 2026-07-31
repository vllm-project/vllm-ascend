#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.attention import parallel_draft_attention as fia_mod
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

FIA_MOD = "vllm_ascend._310p.attention.parallel_draft_attention"

NUM_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 128
NUM_BLOCKS = 8
BATCH = 3
DFLASH_Q = 9  # K + 1
KV_LENS = [200, 133, 65]


def make_vllm_config(*, method="dflash", num_spec=8, arch="DFlashDraftModel", eager=True, tp=2, **hf):
    """`hf` sets extra hf_config fields the scope validator reads.

    Absent ones fall back to getattr defaults, which is what the real
    Qwen3DSpark checkpoint looks like -- it carries none of the DFlash
    diffusion/attention keys.
    """
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            num_speculative_tokens=num_spec,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch], **hf)),
        ),
        model_config=SimpleNamespace(enforce_eager=eager),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


def make_cache(*, dtype=torch.float16, block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS):
    shape = (NUM_BLOCKS, num_kv_heads * HEAD_DIM // 16, block_size, 16)
    return torch.zeros(shape, dtype=dtype)


def make_impl(*, vllm_config=None, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_size=HEAD_DIM, cache=None):
    key_cache = cache if cache is not None else make_cache()
    return SimpleNamespace(
        vllm_config=vllm_config if vllm_config is not None else make_vllm_config(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        scale=HEAD_DIM**-0.5,
        key_cache=key_cache,
        value_cache=key_cache.clone(),
        _fia_scope_validated=False,
    )


def make_metadata(*, q_lens=None, kv_lens=None, block_cols=4, block_dtype=torch.int32):
    q_lens = q_lens if q_lens is not None else [DFLASH_Q] * BATCH
    kv_lens = kv_lens if kv_lens is not None else list(KV_LENS)
    md = SimpleNamespace(
        num_actual_tokens=sum(q_lens),
        seq_lens_list=kv_lens,
        block_tables=torch.zeros(len(kv_lens), block_cols, dtype=block_dtype),
        # Deliberately wrong: the base builder overwrites this field with
        # cumulative endpoints, and the adapter must not read it.
        actual_seq_lengths_q=list(torch.tensor(q_lens).cumsum(0).tolist()),
        causal=False,
        attn_mask=torch.ones(2048, 2048),  # must be ignored
    )
    md.query_lens_cpu = torch.tensor(q_lens, dtype=torch.int32)
    return md


class _FakeFia:
    """Stands in for custom_fused_infer_attention_v310.

    The real op is only registered in a 310P-targeted build, so CI hosts never
    have it. Records the positional tensors alongside the kwargs so the call
    contract can be asserted on either. Like the real op it allocates its own
    output rather than writing into a caller buffer.
    """

    def __init__(self, out_builder=None):
        self.calls = []
        self._out_builder = out_builder

    def __call__(self, query, key, value, **kwargs):
        self.calls.append(dict(kwargs, query=query, key=key, value=value))
        if self._out_builder is not None:
            return self._out_builder(self.calls[-1])
        return torch.zeros_like(query)


def run_forward(impl=None, md=None, fia=None, num_tokens=None):
    impl = impl if impl is not None else make_impl()
    md = md if md is not None else make_metadata()
    fia = fia if fia is not None else _FakeFia()
    n = num_tokens if num_tokens is not None else md.num_actual_tokens
    # Width follows the impl so head-layout cases reshape cleanly and fail on the
    # scope check rather than on the reshape itself.
    width = impl.num_heads * impl.head_size
    query = torch.zeros(n, width, dtype=torch.float16)
    output = torch.zeros(n, width, dtype=torch.float16)
    with (
        mock_patch(f"{FIA_MOD}._fia_op", return_value=fia),
        mock_patch(f"{FIA_MOD}.torch_npu.get_npu_format", return_value=ACL_FORMAT_FRACTAL_NZ),
    ):
        result = fia_mod.forward_parallel_draft_fia(impl, query, md, output)
    return result, fia, output


class TestFiaCallContract(TestBase):
    def test_mask_is_none_regardless_of_metadata(self):
        _, fia, _ = run_forward()
        self.assertIsNone(fia.calls[0]["attn_mask"])

    def test_layout_flags(self):
        _, fia, _ = run_forward()
        kwargs = fia.calls[0]
        self.assertEqual(kwargs["input_layout"], "TND")
        self.assertEqual(kwargs["block_size"], BLOCK_SIZE)
        # The wrapper takes no inner_precise; passing one is a TypeError.
        self.assertNotIn("inner_precise", kwargs)

    def test_every_wrapper_argument_is_passed_explicitly(self):
        """Nothing may fall back to a wrapper default.

        The defaults are upstream's to change -- input_layout's already moved
        from "BSH" to "BSND" -- and a silent move there changes our numerics with
        no test failing anywhere.
        """
        import inspect

        from vllm_ascend._310p.ops.custom_fused_infer_attention import (
            custom_fused_infer_attention_v310,
        )

        accepted = set(inspect.signature(custom_fused_infer_attention_v310).parameters)
        _, fia, _ = run_forward()
        # query/key/value go positionally; the fake records them by name.
        self.assertEqual(accepted, set(fia.calls[0]))

    def test_q_lens_are_raw_not_cumulative(self):
        md = make_metadata()
        _, fia, _ = run_forward(md=md)
        self.assertEqual(fia.calls[0]["actual_seq_lengths_q"], [DFLASH_Q] * BATCH)
        self.assertNotEqual(
            fia.calls[0]["actual_seq_lengths_q"],
            md.actual_seq_lengths_q,
            "adapter used the cumulative endpoints from the base metadata",
        )

    def test_kv_lens_are_passed_through(self):
        _, fia, _ = run_forward()
        self.assertEqual(fia.calls[0]["actual_seq_lengths_kv"], KV_LENS)

    def test_caches_are_passed_by_reference(self):
        """The caches go through untouched: no wrapping, no copy, no reformat."""
        impl = make_impl()
        _, fia, _ = run_forward(impl=impl)
        self.assertIs(fia.calls[0]["key"], impl.key_cache)
        self.assertIs(fia.calls[0]["value"], impl.value_cache)

    def test_head_counts_and_scale(self):
        _, fia, _ = run_forward()
        kwargs = fia.calls[0]
        self.assertEqual(kwargs["num_heads"], NUM_HEADS)
        self.assertEqual(kwargs["num_key_value_heads"], NUM_KV_HEADS)
        self.assertAlmostEqual(kwargs["scale_value"], HEAD_DIM**-0.5)

    def test_query_is_reshaped_to_tnd(self):
        _, fia, _ = run_forward()
        self.assertEqual(tuple(fia.calls[0]["query"].shape), (BATCH * DFLASH_Q, NUM_HEADS, HEAD_DIM))

    def test_result_is_copied_into_the_caller_buffer(self):
        marker = 0.5

        def build_out(kwargs):
            return torch.full_like(kwargs["query"], marker)

        result, _, output = run_forward(fia=_FakeFia(build_out))
        self.assertTrue(bool((output[: BATCH * DFLASH_Q] == marker).all()))
        self.assertIs(result, output, "adapter must return the caller's output buffer")

    def test_dspark_expects_k_queries_not_k_plus_one(self):
        impl = make_impl(vllm_config=make_vllm_config(method="dspark", num_spec=7, arch="Qwen3DSparkModel"))
        md = make_metadata(q_lens=[7] * BATCH)
        _, fia, _ = run_forward(impl=impl, md=md)
        self.assertEqual(fia.calls[0]["actual_seq_lengths_q"], [7] * BATCH)


class TestFiaDynamicGuards(TestBase):
    def test_missing_raw_q_lens_is_refused(self):
        md = make_metadata()
        del md.query_lens_cpu
        with self.assertRaisesRegex(RuntimeError, "query_lens_cpu is missing"):
            run_forward(md=md)

    def test_wrong_q_len_for_method_is_refused(self):
        # DFlash must query K + 1 = 9; 8 would silently drop the anchor.
        md = make_metadata(q_lens=[8] * BATCH)
        with self.assertRaisesRegex(RuntimeError, "expects every request to query 9"):
            run_forward(md=md)

    def test_cumulative_q_lens_are_refused(self):
        md = make_metadata()
        md.query_lens_cpu = torch.tensor([9, 18, 27], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "expects every request to query 9"):
            run_forward(md=md)

    def test_token_count_disagreement_is_refused(self):
        md = make_metadata()
        md.num_actual_tokens = BATCH * DFLASH_Q - 1
        with self.assertRaisesRegex(RuntimeError, "must all agree"):
            run_forward(md=md)

    def test_batch_size_disagreement_is_refused(self):
        md = make_metadata(kv_lens=[200, 133])
        with self.assertRaisesRegex(RuntimeError, "batch size disagreement"):
            run_forward(md=md)

    def test_non_int32_block_table_is_refused(self):
        md = make_metadata(block_dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "rank-2 int32"):
            run_forward(md=md)

    def test_kv_len_exceeding_block_table_capacity_is_refused(self):
        # 2 columns x 128 = 256 addressable, but one request needs 300.
        md = make_metadata(kv_lens=[300, 133, 65], block_cols=2)
        with self.assertRaisesRegex(RuntimeError, "exceeds what its block table can address"):
            run_forward(md=md)

    def test_q_len_greater_than_kv_len_is_refused(self):
        md = make_metadata(kv_lens=[200, 133, 5])
        with self.assertRaisesRegex(RuntimeError, r"need 0 < q_len\(9\) <= kv_len\(5\)"):
            run_forward(md=md)


class TestFiaScopeValidation(TestBase):
    def _expect_refusal(self, pattern, **impl_kwargs):
        with self.assertRaisesRegex(RuntimeError, pattern):
            run_forward(impl=make_impl(**impl_kwargs))

    def test_unsupported_method_is_refused(self):
        # Refused by the adapter's own early check, before the scope validator: it
        # has to resolve queries-per-request from the method first. Reaching here
        # at all means routing let something through, hence the wording.
        self._expect_refusal("reached with unsupported method", vllm_config=make_vllm_config(method="eagle3"))

    def test_scope_validator_refuses_unsupported_method_on_its_own(self):
        """The validator is the documented scope lock, so its method branch is
        covered directly -- the adapter's early check shadows it in the forward
        path, which would otherwise leave it untested."""
        cache = make_cache()
        with self.assertRaisesRegex(RuntimeError, "only covers"):
            fia_mod.validate_fia_scope(
                vllm_config=make_vllm_config(method="eagle3"),
                query=torch.zeros(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16),
                key_cache=cache,
                value_cache=cache.clone(),
                num_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_size=HEAD_DIM,
            )

    def test_unexpected_k_is_refused(self):
        self._expect_refusal("must be >= 1", vllm_config=make_vllm_config(num_spec=0))

    def test_unsupported_architecture_is_refused(self):
        self._expect_refusal("outside this scope", vllm_config=make_vllm_config(arch="LlamaForCausalLM"))

    def test_target_in_graph_mode_is_allowed(self):
        """model_config.enforce_eager describes the engine, not this route.

        "Target in ACLGraph, drafter eager" is the configuration we want, so an
        engine-wide eager requirement here would reject it for a constraint that
        only applies to this op. What must not happen is captured separately, in
        TestFiaCaptureGuard.
        """
        _, fia, _ = run_forward(impl=make_impl(vllm_config=make_vllm_config(eager=False)))
        self.assertEqual(len(fia.calls), 1)

    def test_tp4_layout_is_allowed(self):
        """TP only shards heads; the per-rank layout is what is checked. Qwen3-8B
        at TP=4 is 8 query / 2 KV heads, which satisfies the structural rules, so
        it must run rather than be refused."""
        impl = make_impl(
            vllm_config=make_vllm_config(tp=4),
            num_heads=8,
            num_kv_heads=2,
            cache=make_cache(num_kv_heads=2),
        )
        _, fia, _ = run_forward(impl=impl)
        self.assertEqual(len(fia.calls), 1)
        self.assertEqual(fia.calls[0]["num_heads"], 8)
        self.assertEqual(fia.calls[0]["num_key_value_heads"], 2)

    def test_too_many_per_rank_heads_is_refused(self):
        """Qwen3-8B is 32 query heads, so TP=1 lands outside the envelope the
        operator's own test covers (1..16). Refuse rather than run untested."""
        self._expect_refusal(
            "exceeds the 16 this scope covers",
            num_heads=32,
            num_kv_heads=8,
            cache=make_cache(num_kv_heads=8),
        )

    def test_illegal_gqa_is_refused(self):
        # 17 query heads do not divide evenly by 4 KV heads.
        self._expect_refusal("invalid GQA layout", num_heads=17)

    def test_non_16_aligned_kv_is_refused(self):
        # Any num_kv_heads * 128 is 16-aligned, so break it with a small head_dim:
        # 1 KV head * 8 = 8 is not a multiple of 16.
        self._expect_refusal(
            "must be a multiple of 16",
            num_heads=4,
            num_kv_heads=1,
            head_size=8,
            cache=make_cache(num_kv_heads=1),
        )

    def test_bf16_is_refused(self):
        self._expect_refusal("only supports float16", cache=make_cache(dtype=torch.bfloat16))

    def test_wrong_cache_block_size_is_refused(self):
        self._expect_refusal("this scope only covers 128", cache=make_cache(block_size=64))

    def test_non_nz_cache_format_is_refused(self):
        impl = make_impl()
        md = make_metadata()
        query = torch.zeros(md.num_actual_tokens, NUM_HEADS * HEAD_DIM, dtype=torch.float16)
        output = torch.zeros_like(query)
        with (
            mock_patch(f"{FIA_MOD}._fia_op", return_value=_FakeFia()),
            mock_patch(f"{FIA_MOD}.torch_npu.get_npu_format", return_value=2),
        ):
            with self.assertRaisesRegex(RuntimeError, "expected ACL_FORMAT_FRACTAL_NZ"):
                fia_mod.forward_parallel_draft_fia(impl, query, md, output)

    def test_scope_is_validated_once_then_cached(self):
        impl = make_impl()
        fia = _FakeFia()
        with mock_patch(f"{FIA_MOD}.validate_fia_scope") as validator:
            with (
                mock_patch(f"{FIA_MOD}._fia_op", return_value=fia),
                mock_patch(f"{FIA_MOD}.torch_npu.get_npu_format", return_value=ACL_FORMAT_FRACTAL_NZ),
            ):
                for _ in range(3):
                    md = make_metadata()
                    query = torch.zeros(md.num_actual_tokens, NUM_HEADS * HEAD_DIM, dtype=torch.float16)
                    fia_mod.forward_parallel_draft_fia(impl, query, md, torch.zeros_like(query))
        self.assertEqual(validator.call_count, 1, "startup invariants must not be re-checked per step")
        self.assertEqual(len(fia.calls), 3)


class TestFiaOpResolution(TestBase):
    def test_missing_op_fails_loud_without_fallback(self):
        # A wheel built for a non-310P SOC imports fine but carries no such
        # symbol. That must stop the run, not silently reroute to the causal
        # split-fuse kernel.
        # Swap the whole namespace for an empty one rather than patching the
        # attribute: torch.ops._C_ascend resolves lazily, so the attribute may
        # legitimately not exist yet and patch.object would fail on setup.
        with mock_patch.object(torch.ops, "_C_ascend", SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                fia_mod._fia_op()


class TestFiaReturnHandling(TestBase):
    def test_mismatched_return_shape_is_refused(self):
        def bad_shape(call):
            q = call["query"]
            # Same element count, wrong layout -- numel alone would accept this.
            return torch.zeros(q.shape[0], q.shape[2], q.shape[1], dtype=q.dtype)

        with self.assertRaisesRegex(RuntimeError, "expected the query shape"):
            run_forward(fia=_FakeFia(bad_shape))

    def test_mismatched_return_dtype_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "expected the query shape"):
            run_forward(fia=_FakeFia(lambda call: torch.zeros_like(call["query"], dtype=torch.float32)))


class TestFiaNonCausalScopeLocks(TestBase):
    """This route sends attn_mask=None, which the kernel maps to NO_MASK.

    That is only correct for a drafter whose attention is full and non-causal.
    A causal or sliding-window checkpoint would still run and return plausible
    numbers, so each of these has to be refused at startup rather than found
    later as an accuracy loss.

    The public z-lab/Qwen3-8B-DFlash-b16 checkpoint satisfies all of them:
    use_sliding_window False, sliding_window None, five full_attention layers.
    """

    def _refuse(self, pattern, **hf):
        with self.assertRaisesRegex(RuntimeError, pattern):
            run_forward(impl=make_impl(vllm_config=make_vllm_config(**hf)))

    def test_sliding_window_layer_is_refused(self):
        self._refuse(
            "other than full_attention",
            layer_types=["full_attention", "sliding_attention"],
        )

    def test_use_sliding_window_flag_is_refused(self):
        self._refuse("sliding-window attention", use_sliding_window=True)

    def test_sliding_window_size_is_refused(self):
        self._refuse("sliding-window attention", sliding_window=4096)

    def test_causal_dflash_config_is_refused(self):
        self._refuse("causal attention", dflash_config=SimpleNamespace(causal=True))

    def test_all_full_attention_is_accepted(self):
        # The real checkpoint's layer_types; must not be caught by the above.
        _, fia, _ = run_forward(
            impl=make_impl(vllm_config=make_vllm_config(layer_types=["full_attention"] * 5))
        )
        self.assertEqual(len(fia.calls), 1)


class TestFiaSpeculativeTokenScope(TestBase):
    """K is not a fixed list: the kernel tiles the query dimension, so the only
    real ceiling is the draft checkpoint's own diffusion block."""

    def test_k15_is_accepted_for_dflash(self):
        """The public checkpoint's recommended K: block_size 16, so q = 16."""
        impl = make_impl(vllm_config=make_vllm_config(num_spec=15, block_size=16))
        _, fia, _ = run_forward(impl=impl, md=make_metadata(q_lens=[16] * BATCH))
        self.assertEqual(fia.calls[0]["actual_seq_lengths_q"], [16] * BATCH)

    def test_k8_is_accepted_for_dflash(self):
        """Smaller than the block allows is legitimate -- the Ascend MRV1 DFlash
        tests use K=8 against the same block-16 checkpoint."""
        impl = make_impl(vllm_config=make_vllm_config(num_spec=8, block_size=16))
        _, fia, _ = run_forward(impl=impl, md=make_metadata(q_lens=[9] * BATCH))
        self.assertEqual(fia.calls[0]["actual_seq_lengths_q"], [9] * BATCH)

    def test_dflash_loses_one_row_to_the_anchor(self):
        """K=16 on a block-16 checkpoint needs 17 rows; DSpark's K=16 would fit."""
        impl = make_impl(vllm_config=make_vllm_config(num_spec=16, block_size=16))
        with self.assertRaisesRegex(RuntimeError, "diffusion block holds"):
            run_forward(impl=impl, md=make_metadata(q_lens=[17] * BATCH))

    def test_dspark_uses_the_whole_block(self):
        impl = make_impl(
            vllm_config=make_vllm_config(
                method="dspark", num_spec=7, arch="Qwen3DSparkModel", block_size=7
            )
        )
        _, fia, _ = run_forward(impl=impl, md=make_metadata(q_lens=[7] * BATCH))
        self.assertEqual(fia.calls[0]["actual_seq_lengths_q"], [7] * BATCH)

    def test_dspark_beyond_its_block_is_refused(self):
        impl = make_impl(
            vllm_config=make_vllm_config(
                method="dspark", num_spec=8, arch="Qwen3DSparkModel", block_size=7
            )
        )
        with self.assertRaisesRegex(RuntimeError, "diffusion block holds"):
            run_forward(impl=impl, md=make_metadata(q_lens=[8] * BATCH))

    def test_q_beyond_one_kernel_tile_is_allowed(self):
        """seqStepQ is 32 at head_dim <= 128 and the host splits into
        ceil(q / seqStepQ) blocks, so a query longer than one tile is tiled, not
        rejected. Guards against reintroducing a ceiling the kernel never had."""
        impl = make_impl(vllm_config=make_vllm_config(num_spec=39, block_size=40))
        _, fia, _ = run_forward(impl=impl, md=make_metadata(q_lens=[40] * BATCH, block_cols=8))
        self.assertEqual(fia.calls[0]["actual_seq_lengths_q"], [40] * BATCH)

    def test_scope_is_checked_before_query_length(self):
        """An out-of-scope configuration must be reported as such.

        The q-len check assumes the configuration is in scope, so when it ran
        first an out-of-scope K surfaced as a wrong q-len and blamed the metadata
        for carrying cumulative endpoints.
        """
        impl = make_impl(vllm_config=make_vllm_config(num_spec=16, block_size=16))
        with self.assertRaisesRegex(RuntimeError, "diffusion block holds"):
            run_forward(impl=impl, md=make_metadata(q_lens=[9] * BATCH))


class TestFiaCaptureGuard(TestBase):
    """This op must never end up inside an ACLGraph capture.

    Its actual_seq_lengths_q / _kv are host SymInt[] read at dispatch, so a
    captured graph would freeze one step's lengths and replay them for every
    later step -- plausible numbers, silently wrong. The guard fires at the
    moment it would happen rather than inferring from configuration, so the
    supported split (target captured, drafter eager) stays legal.
    """

    def _run_with_ctx(self, *, available, capturing):
        ctx = SimpleNamespace(capturing=capturing)
        with (
            mock_patch(f"{FIA_MOD}.is_forward_context_available", return_value=available),
            mock_patch(f"{FIA_MOD}._EXTRA_CTX", ctx),
        ):
            return run_forward()

    def test_capture_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "during ACLGraph capture"):
            self._run_with_ctx(available=True, capturing=True)

    def test_replay_is_allowed(self):
        """Replay sets capturing False; only capture is the problem."""
        _, fia, _ = self._run_with_ctx(available=True, capturing=False)
        self.assertEqual(len(fia.calls), 1)

    def test_extra_ctx_is_not_read_without_a_forward_context(self):
        """Reading _EXTRA_CTX with no forward context raises "Forward context is
        not set", so the availability check has to short-circuit. Locked with an
        object that detonates on any attribute access."""

        class _Detonator:
            def __getattr__(self, name):
                raise AssertionError(f"_EXTRA_CTX.{name} read without a forward context")

        with (
            mock_patch(f"{FIA_MOD}.is_forward_context_available", return_value=False),
            mock_patch(f"{FIA_MOD}._EXTRA_CTX", _Detonator()),
        ):
            _, fia, _ = run_forward()
        self.assertEqual(len(fia.calls), 1)
