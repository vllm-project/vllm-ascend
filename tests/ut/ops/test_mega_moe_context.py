# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for target/draft sharing without native collectives."""

import sys
import unittest
from inspect import signature
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_ascend.ops.fused_moe.mega_moe_context import get_main_owned_buffer, is_draft_moe_config


def _buffer_factory(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    *,
    max_recv_token_num=0,
    dispatch_quant_mode=0,
    dispatch_quant_out_dtype=None,
    combine_quant_mode=0,
    comm_alg="",
    topk_weights_type=0,
):
    return SimpleNamespace(**locals(), context=object(), _ctx_manager=object(), ep_world_size=8, ccl_buffer_size=4096)


class TestMainOwnedMegaMoeBuffer(unittest.TestCase):
    def setUp(self):
        self.scope = SimpleNamespace()
        self.group = object()
        self.factory = Mock(wraps=_buffer_factory)
        # Preserve the CANN factory's signature and module for ABI binding.
        self.factory.__signature__ = signature(_buffer_factory)
        self.factory.__module__ = __name__
        self.size_query = Mock(return_value=4096)
        self.size_patch = patch.object(
            sys.modules[__name__], "_get_mega_moe_ccl_buffer_size", self.size_query, create=True
        )
        self.size_patch.start()
        self.addCleanup(self.size_patch.stop)

    def _get_buffer(self, is_draft=False, **overrides):
        params = dict(
            group=self.group,
            num_experts=384,
            num_max_tokens_per_rank=128,
            num_topk=6,
            hidden=16,
            intermediate_hidden=64,
            max_recv_token_num=1024,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype="float8",
        )
        params.update(overrides)
        return get_main_owned_buffer(self.scope, self.factory, is_draft=is_draft, **params)

    def test_main_draft_main_uses_one_context_with_independent_metadata(self):
        main = self._get_buffer()
        draft = self._get_buffer(is_draft=True, num_experts=128, num_topk=3, intermediate_hidden=32)
        another_main = self._get_buffer()

        self.factory.assert_called_once()
        for view in (draft, another_main):
            self.assertIs(view.context, main.context)
            self.assertIs(view._ctx_manager, main._ctx_manager)
            self.assertIs(view.group, main.group)
            self.assertEqual(view.ccl_buffer_size, main.ccl_buffer_size)
        self.assertEqual((main.num_experts, main.num_topk, main.intermediate_hidden), (384, 6, 64))
        self.assertEqual((draft.num_experts, draft.num_topk, draft.intermediate_hidden), (128, 3, 32))
        draft.dispatch_quant_mode = 0
        self.assertEqual(main.dispatch_quant_mode, 4)
        self.assertEqual(another_main.dispatch_quant_mode, 4)
        self.assertEqual(self.size_query.call_args_list[0].args[:4], (8, 128, 128, 3))

    def test_draft_first_fails_without_allocating_a_context(self):
        with self.assertRaisesRegex(RuntimeError, "Main must initialize"):
            self._get_buffer(is_draft=True)
        self.factory.assert_not_called()
        self.assertFalse(hasattr(self.scope, "_main_mega_moe_symm_buffer"))

    def test_separate_mc2_groups_have_separate_owners(self):
        main = self._get_buffer()
        self.scope = SimpleNamespace()
        self.group = object()
        other = self._get_buffer()
        self.assertIsNot(main.context, other.context)
        self.assertEqual(self.factory.call_count, 2)

    def test_incompatible_group_or_communication_settings_do_not_allocate(self):
        self._get_buffer()
        for name, value in (
            ("group", object()),
            ("hidden", 32),
            ("dispatch_quant_mode", 0),
            ("dispatch_quant_out_dtype", None),
            ("combine_quant_mode", 4),
            ("comm_alg", "hierarchy"),
            ("topk_weights_type", 1),
        ):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self._get_buffer(is_draft=True, **{name: value})
        self.factory.assert_called_once()
        self.size_query.assert_not_called()

    def test_draft_token_capacity_cannot_exceed_main(self):
        self._get_buffer()
        with self.assertRaisesRegex(RuntimeError, "token capacity"):
            self._get_buffer(is_draft=True, num_max_tokens_per_rank=129)
        self.factory.assert_called_once()

    def test_draft_ccl_capacity_cannot_exceed_main(self):
        self._get_buffer()
        self.size_query.return_value = 4097
        with self.assertRaisesRegex(RuntimeError, "exceeds main"):
            self._get_buffer(is_draft=True)
        self.factory.assert_called_once()


class TestDraftMoEConfig(unittest.TestCase):
    def _config(self, runner_type="generate", target_experts=384, draft_experts=128):
        return SimpleNamespace(
            model_config=SimpleNamespace(runner_type=runner_type, get_num_experts=lambda: target_experts),
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(get_num_experts=lambda: draft_experts)
            ),
        )

    def test_eager_mrv2_draft_is_identified_under_target_config(self):
        config = self._config()
        self.assertTrue(is_draft_moe_config(SimpleNamespace(num_experts=128), config))
        self.assertFalse(is_draft_moe_config(SimpleNamespace(num_experts=384), config))

    def test_redundant_experts_do_not_change_model_role(self):
        config = self._config()
        self.assertFalse(is_draft_moe_config(SimpleNamespace(num_experts=400, global_redundant_expert_num=16), config))
        self.assertTrue(is_draft_moe_config(SimpleNamespace(num_experts=144, global_redundant_expert_num=16), config))

    def test_explicit_draft_runner(self):
        self.assertTrue(is_draft_moe_config(SimpleNamespace(num_experts=128), self._config(runner_type="draft")))

    def test_no_speculation_or_same_expert_count(self):
        moe = SimpleNamespace(num_experts=384)
        config = self._config(draft_experts=384)
        self.assertFalse(is_draft_moe_config(moe, config))
        config.model_config.runner_type = "draft"
        self.assertFalse(is_draft_moe_config(moe, config))
        config.model_config.runner_type = "generate"
        config.speculative_config = None
        self.assertFalse(is_draft_moe_config(moe, config))
