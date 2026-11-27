# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C2 branch coverage and shared warmup/runtime argument binding."""

import ast
import inspect
import math
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from tests.ut.model_executor.warmup.helpers import make_registry
from tests.ut.ops.test_indexer_k_fusion import _load_launcher


class TestC2PoolWarmup(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_launcher("vllm_ascend/ops/triton/compressor/compressor_triton.py")

    def setUp(self):
        self.owner = self.module.C2PoolKernel()

    def test_keys_cover_pcp_norm_and_weight_signatures(self):
        keys = self.owner.get_warmup_keys(head_dim=512)
        self.assertEqual(len(keys), 6)
        self.assertEqual(len(set(keys)), 6)
        self.assertEqual(
            {(key.has_prev_rows, key.fuse_token_norm) for key in keys},
            {(pcp, norm) for pcp in (False, True) for norm in (False, True)},
        )
        for key in keys:
            inputs = self.owner.warmup_inputs(key)
            self.assertEqual(inputs["prev_rows"] is not None, key.has_prev_rows)
            self.assertEqual(inputs["prev_row_indices"] is not None, key.has_prev_rows)
            self.assertEqual(inputs["norm_w"].dtype, key.norm_dtype)
            if not key.fuse_token_norm:
                self.assertEqual(key.norm_dtype, torch.bfloat16)
            self.assertEqual(self.owner.compile_key(vars(key)), key)

    def test_compile_binds_every_argument_without_execution(self):
        raw_kernel = type(self.owner).kernel
        kernel = MagicMock(arg_names=raw_kernel.arg_names)
        keys = self.owner.get_warmup_keys(head_dim=512)
        with patch.object(self.owner, "kernel", kernel):
            for key in keys:
                self.owner.compile(key)
                kwargs = dict(kernel.warmup.call_args.kwargs)
                self.assertEqual(kwargs.pop("grid"), (1,))
                bound = inspect.signature(raw_kernel).bind(**kwargs).arguments
                self.assertEqual(bound["HEAD_DIM"], 512)
                self.assertEqual(bound["HAS_PREV_ROWS"], key.has_prev_rows)
                self.assertEqual(bound["FUSE_TOKEN_NORM"], key.fuse_token_norm)
                self.assertEqual(bound["norm_w_ptr"].dtype, key.norm_dtype)
                self.assertEqual(bound["prev_rows_ptr"] is not None, key.has_prev_rows)
                self.assertIsInstance(bound["rms_eps"], float)
                self.assertEqual(bound["EPS"], 0.0)
                self.assertTrue(bound["TOKEN_ALIGNED"])
                self.assertEqual(bound["RATIO"], 2)
        self.assertEqual(kernel.warmup.call_count, len(keys))
        kernel.__getitem__.assert_not_called()
        self.assertFalse(self.owner._warming)

    def test_registration_window_and_deduplication(self):
        registry = make_registry()
        self.owner.register_warmup(head_dim=512)
        self.assertEqual(len(registry), 0)
        with registry.activate():
            self.owner.register_warmup(head_dim=512)
            self.owner.register_warmup(head_dim=512)
        self.assertEqual(len(registry), 1)
        with (
            patch.object(self.owner, "compile") as compile_key,
            patch("vllm.distributed.is_global_first_rank", return_value=False),
        ):
            registry.warmup()
        self.assertEqual(compile_key.call_count, 6)

    def test_runtime_uses_actual_grid_and_dynamic_request_bounds(self):
        key = self.owner.get_warmup_keys(head_dim=512)[-1]
        inputs = self.owner.warmup_inputs(key)
        inputs.update(total_group_slots=51, group_slots_per_batch=3, num_batch=17, num_cache_blocks=65537, grid_size=8)
        kernel = MagicMock(arg_names=self.owner.kernel.arg_names)
        with patch.object(self.owner, "kernel", kernel):
            self.owner(**inputs)
        kernel.__getitem__.assert_called_once_with((8,))
        kwargs = kernel.__getitem__.return_value.call_args.kwargs
        self.assertEqual(kwargs["total_group_slots"], 51)
        self.assertEqual(kwargs["group_slots_per_batch"], 3)
        self.assertEqual(kwargs["num_batch"], 17)
        self.assertEqual(kwargs["num_cache_blocks"], 65537)
        kernel.warmup.assert_not_called()

    def test_fused_expression_uses_runtime_epsilon_not_legacy_constant(self):
        # Evaluate the actual fused expression with scalar stand-ins. This
        # catches the rebased kernel's stale `eps` name without emulating NPU
        # arithmetic or replacing the production expression with a reference.
        tree = ast.parse(inspect.getsource(type(self.owner).kernel))
        expressions = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "pooled" for target in node.targets)
            and any(isinstance(call, ast.Attribute) and call.attr == "rsqrt" for call in ast.walk(node.value))
        ]
        self.assertEqual(len(expressions), 1)
        expression = compile(ast.Expression(expressions[0]), "<fused-c2-rmsnorm>", "eval")
        for epsilon in (0.25, 1.0, 4.0):
            values = dict(pooled_bf16=2.0, mean_square=4.0, norm_w=3.0, rms_eps=epsilon, EPS=0.0)
            actual = eval(expression, {"tl": SimpleNamespace(rsqrt=lambda value: 1.0 / math.sqrt(value))}, values)
            self.assertAlmostEqual(actual, 6.0 / math.sqrt(4.0 + epsilon))

    def test_public_entry_preserves_pool_before_ring_write_and_pcp_ownership(self):
        device = SimpleNamespace(type="npu")

        def tensor(shape, dtype):
            return MagicMock(shape=shape, ndim=len(shape), dtype=dtype, device=device)

        for key in self.owner.get_warmup_keys(head_dim=512):
            with self.subTest(key=key):
                events = []
                out = tensor((17, 512), torch.bfloat16)
                out.zero_.side_effect = lambda: events.append("zero")
                metadata = tensor((5, 3), torch.int32)
                norm = tensor((512,), key.norm_dtype) if key.fuse_token_norm else None
                pool = MagicMock(arg_names=self.owner.kernel.arg_names)
                pool.__getitem__.return_value.side_effect = lambda **kwargs: events.append("pool")
                write = MagicMock()
                write.__getitem__.return_value.side_effect = lambda *args, **kwargs: events.append("write")
                owner = self.module._C2_POOL_KERNEL
                with patch.object(owner, "kernel", pool), patch.object(self.module, "_cache_update_kernel", write):
                    result = self.module.compressor_from_projected(
                        tensor((17, 512), torch.float32),
                        tensor((17, 512), torch.float32),
                        tensor((8, 32, 1024), torch.float32),
                        metadata,
                        out,
                        max_query_len=17,
                        num_cores=8,
                        prev_rows=tensor((3, 1024), torch.float32) if key.has_prev_rows else None,
                        prev_row_indices=tensor((3,), torch.int64) if key.has_prev_rows else None,
                        norm_w=norm,
                        rms_eps=1e-6,
                    )
                self.assertIs(result, out)
                self.assertEqual(events, ["zero", "pool"] if key.has_prev_rows else ["zero", "pool", "write"])
                kwargs = pool.__getitem__.return_value.call_args.kwargs
                self.assertIs(kwargs["norm_w_ptr"], norm if norm is not None else out)
                self.assertIs(kwargs["block_table_ptr"], metadata[4])
                self.assertEqual(kwargs["HAS_PREV_ROWS"], key.has_prev_rows)
                self.assertEqual(kwargs["FUSE_TOKEN_NORM"], key.fuse_token_norm)
                self.assertEqual(kwargs["EPS"], 0.0)
                self.assertEqual(kwargs["rms_eps"], 1e-6)
