# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical contract test runnable without importing the NPU runtime."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F


def load_sparse_wrapper(adaptor_name="BaseDeviceAdaptor"):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/device/device_op.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    adaptor = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "BaseDeviceAdaptor")
    methods = {"execute_sparse_flash_attention_process", "_execute_kv_quant_sparse_flash_attention_vllm"}
    adaptor.body = [n for n in adaptor.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    classes: list[ast.stmt] = [adaptor]
    if adaptor_name != "BaseDeviceAdaptor":
        a5 = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == adaptor_name)
        a5.body = [n for n in a5.body if isinstance(n, ast.FunctionDef) and n.name in methods] or [ast.Pass()]
        classes.append(a5)
    namespace = {
        "torch": torch,
        "torch_npu": SimpleNamespace(),
        "F": F,
        "_SFA_MAX_GENERIC_QUERY_GROUP_SIZE": 64,
        "_SFA_PADDED_QUERY_GROUP_SIZE": 128,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), str(source), "exec"), namespace)
    return namespace[adaptor_name]


class TestQuantizedSparseAttention(unittest.TestCase):
    def test_quantized_qsfa_routes_by_model_and_rope(self):
        for adaptor_name in ("BaseDeviceAdaptor", "A5DeviceAdaptor"):
            adaptor = load_sparse_wrapper(adaptor_name)
            runtime = adaptor._execute_kv_quant_sparse_flash_attention_vllm.__globals__["torch_npu"]
            for kimi, rope_dim, lse, packed, custom, custom_lse in (
                (False, 64, False, False, False, None),
                (False, 64, True, False, True, True),
                (False, 0, False, False, True, False),
                (True, 64, False, False, True, True),
                (True, 0, False, False, True, True),
                (True, 64, False, True, True, True),
            ):
                with self.subTest(adaptor=adaptor_name, kimi=kimi, rope=rope_dim, lse=lse, packed=packed):
                    q = torch.randn(1, 2, 4)
                    rope = torch.randn(1, 2, rope_dim) if rope_dim else None
                    packed_q = torch.cat((q, rope), dim=-1) if packed else None
                    cache = torch.empty(1, 3, 1, 4, dtype=torch.int8)
                    with (
                        patch.object(
                            torch.ops._C_ascend, "npu_kv_quant_sparse_flash_attention_vllm", create=True
                        ) as op,
                        patch.object(runtime, "npu_kv_quant_sparse_flash_attention", create=True) as builtin,
                    ):
                        adaptor._execute_kv_quant_sparse_flash_attention_vllm(
                            SimpleNamespace(scale=0.5, qk_rope_head_dim=rope_dim, is_kimi_k31=kimi),
                            q,
                            rope,
                            cache,
                            torch.zeros(1, 1),
                            torch.zeros(1, 1, 1),
                            torch.tensor([1]),
                            torch.tensor([3]),
                            return_lse=lse,
                            packed_query=packed_q,
                        )
                    self.assertEqual(op.called, custom)
                    self.assertEqual(builtin.called, not custom)
                    if custom:
                        self.assertEqual(op.call_args.kwargs["return_softmax_lse"], custom_lse)

    def test_a5_fp8_keeps_native_heads(self):
        adaptor = load_sparse_wrapper("A5DeviceAdaptor")
        q = torch.randn(1, 96, 4)
        rope = torch.empty(1, 96, 0)
        cache = torch.empty(1, 3, 1, 4, dtype=torch.float8_e4m3fn)
        empty_rope_pads = []
        original_pad = F.pad

        def tracked_pad(tensor, padding):
            if tensor.shape[-1] == 0:
                empty_rope_pads.append(tuple(tensor.shape))
            return original_pad(tensor, padding)

        with (
            patch.object(F, "pad", side_effect=tracked_pad),
            patch.object(
                adaptor,
                "_execute_kv_quant_sparse_flash_attention_vllm",
                create=True,
                side_effect=lambda impl, query, *args, **kwargs: query,
            ),
        ):
            output = adaptor.execute_sparse_flash_attention_process(
                SimpleNamespace(scale=0.5, is_kimi_k31=True),
                q,
                rope,
                (cache,),
                torch.empty(1, 1, 3),
                SimpleNamespace(block_table=torch.zeros(1, 1)),
                torch.tensor([1]),
                torch.tensor([3]),
            )
        self.assertEqual(len(empty_rope_pads), 0)
        torch.testing.assert_close(output, q)

    def test_a5_int8_keeps_native_heads(self):
        adaptor = load_sparse_wrapper("A5DeviceAdaptor")
        for group in (6, 12, 48, 96):
            query = torch.randn(1, group, 4)
            rope = torch.empty(1, group, 0)
            cache = torch.empty(1, 3, 1, 4, dtype=torch.int8)

            def native(impl, q, qr, *args, _query=query, _rope=rope, **kwargs):
                self.assertIs(q, _query)
                self.assertIs(qr, _rope)
                return q

            with patch.object(
                adaptor, "_execute_kv_quant_sparse_flash_attention_vllm", create=True, side_effect=native
            ):
                output = adaptor.execute_sparse_flash_attention_process(
                    SimpleNamespace(scale=0.5, is_kimi_k31=True),
                    query,
                    rope,
                    (cache,),
                    torch.empty(1, 1, 3),
                    SimpleNamespace(block_table=torch.zeros(1, 1)),
                    torch.tensor([1]),
                    torch.tensor([3]),
                )
            self.assertIs(output, query)

    def test_int8_attention_and_lse_preserve_each_kv_group(self):
        # PR #54 supports native query groups 1..128 on A3 and A5.
        adaptor = load_sparse_wrapper()
        for group in (1, 6, 12, 48, 64, 65, 95, 96, 97, 128):
            for kv_heads in (1, 2):
                for rope_dim in (0, 2):
                    for return_lse in (False, True):
                        with self.subTest(group=group, kv_heads=kv_heads, rope_dim=rope_dim, return_lse=return_lse):
                            self._check_case(adaptor, group, kv_heads, rope_dim, return_lse)

    def test_a3_a5_native_heads_and_rope0_routing(self):
        for adaptor_name, dtype in (
            ("BaseDeviceAdaptor", torch.int8),
            ("A5DeviceAdaptor", torch.int8),
            ("A5DeviceAdaptor", torch.float8_e4m3fn),
        ):
            adaptor = load_sparse_wrapper(adaptor_name)
            for group in range(1, 129):
                for rope_dim, return_lse in ((0, False), (0, True), (2, True), (2, False)):
                    query = torch.randn(1, group, 4)
                    rope = torch.randn(1, group, rope_dim)
                    cache = torch.empty(1, 3, 1, 4, dtype=dtype)

                    def native(*, _query=query, _rope=rope, _group=group, **kwargs):
                        self.assertTrue(kwargs["return_softmax_lse"])
                        expected = torch.cat((_query, _rope), dim=-1)
                        torch.testing.assert_close(kwargs["query"], expected)
                        output = kwargs["query"][..., :4]
                        return output, torch.zeros(1, 1, _group), torch.ones(1, 1, _group)

                    runtime = adaptor._execute_kv_quant_sparse_flash_attention_vllm.__globals__["torch_npu"]
                    with (
                        patch.object(
                            torch.ops._C_ascend,
                            "npu_kv_quant_sparse_flash_attention_vllm",
                            create=True,
                            side_effect=native,
                        ),
                        patch.object(
                            runtime,
                            "npu_kv_quant_sparse_flash_attention",
                            create=True,
                            side_effect=AssertionError("A5 must use the custom operator"),
                        ),
                    ):
                        result = adaptor._execute_kv_quant_sparse_flash_attention_vllm(
                            SimpleNamespace(scale=0.5, is_kimi_k31=True, qk_rope_head_dim=rope_dim),
                            query,
                            rope,
                            cache,
                            torch.zeros(1, 1),
                            torch.empty(1, 1, 3),
                            torch.tensor([1]),
                            torch.tensor([3]),
                            return_lse=return_lse,
                        )
                    output = result[0] if isinstance(result, tuple) else result
                    torch.testing.assert_close(output, query)

    def test_a5_attention_and_lse_preserve_native_heads(self):
        adaptor = load_sparse_wrapper("A5DeviceAdaptor")
        for group in (12, 24, 65, 95, 96, 97, 128):
            for kv_heads in (1, 2):
                for rope_dim in (0, 64):
                    for return_lse in (False, True):
                        with self.subTest(group=group, kv_heads=kv_heads, rope_dim=rope_dim, return_lse=return_lse):
                            self._check_case(
                                adaptor,
                                group,
                                kv_heads,
                                rope_dim,
                                return_lse,
                                dtype=torch.float8_e4m3fn,
                            )

    def test_a5_packed_query_preserves_order_without_cat(self):
        adaptor = load_sparse_wrapper("A5DeviceAdaptor")
        for heads in (12, 24, 65, 95, 96, 97, 128):
            for rope_dim in (0, 64):
                query = torch.arange(3 * heads * (4 + rope_dim)).reshape(3, heads, 4 + rope_dim).float()
                packed = query.transpose(0, 1).contiguous().transpose(0, 1)

                def native(*, _heads=heads, _rope_dim=rope_dim, _query=query, **kwargs):
                    q = kwargs["query"]
                    self.assertTrue(q.is_contiguous())
                    self.assertEqual(q.shape, (3, _heads, 4 + _rope_dim))
                    torch.testing.assert_close(q, _query)
                    return q[..., :4], q[..., 0].unsqueeze(0), q[..., 1].unsqueeze(0)

                with (
                    patch.object(
                        torch.ops._C_ascend, "npu_kv_quant_sparse_flash_attention_vllm", create=True, side_effect=native
                    ),
                    patch.object(F, "pad", side_effect=AssertionError("A5 must not pad query heads")),
                    patch("torch.cat", side_effect=AssertionError("packed query must bypass cat")),
                ):
                    result = adaptor.execute_sparse_flash_attention_process(
                        SimpleNamespace(scale=0.5, is_kimi_k31=True, qk_rope_head_dim=rope_dim),
                        None,
                        None,
                        (torch.empty(1, 128, 1, 16, dtype=torch.float8_e4m3fn),),
                        None,
                        SimpleNamespace(block_table=None),
                        None,
                        None,
                        return_lse=True,
                        packed_query=packed,
                    )
                for actual, expected in zip(
                    result, (query[..., :4], query[..., 0].unsqueeze(0), query[..., 1].unsqueeze(0))
                ):
                    torch.testing.assert_close(actual, expected)

    def test_a3_packed_query_preserves_group_precision_without_cat(self):
        adaptor = load_sparse_wrapper()
        for group in (6, 12, 64, 96, 128):
            for kv_heads in (1, 2):
                for rope_dim in (0, 64):
                    for return_lse in (False, True):
                        with self.subTest(group=group, kv_heads=kv_heads, rope_dim=rope_dim, return_lse=return_lse):
                            self._check_case(adaptor, group, kv_heads, rope_dim, return_lse, packed=True)

    def test_packed_query_rejects_non_quantized_cache(self):
        with self.assertRaisesRegex(ValueError, "quantized SFA"):
            load_sparse_wrapper().execute_sparse_flash_attention_process(
                None,
                None,
                None,
                (torch.empty(1, dtype=torch.bfloat16),),
                None,
                SimpleNamespace(block_table=None),
                None,
                None,
                packed_query=torch.empty(1, 96, 4),
            )

    def _check_case(
        self,
        adaptor,
        group,
        kv_heads,
        rope_dim,
        return_lse,
        *,
        dtype=torch.int8,
        packed=False,
    ):
        generator = torch.Generator().manual_seed(17)
        q = torch.randn(2, kv_heads * group, 4, generator=generator)
        rope = torch.randn(2, kv_heads * group, rope_dim, generator=generator)
        key = torch.randint(-4, 5, (1, 3, kv_heads, 4), generator=generator, dtype=torch.int8).to(dtype)
        key_rope = torch.randn(3, kv_heads, rope_dim, generator=generator)
        table = torch.zeros(1, 1, dtype=torch.int32)
        topk = torch.arange(3, dtype=torch.int32).view(1, 1, 3).expand(2, 1, 3)

        def reference(query, query_rope):
            scores = torch.einsum("tkhd,skd->tkhs", query.unflatten(1, (kv_heads, -1)), key[0].float())
            scores += torch.einsum("tkhd,skd->tkhs", query_rope.unflatten(1, (kv_heads, -1)), key_rope)
            scores *= 0.5
            maximum = scores.amax(-1)
            total = torch.exp(scores - maximum.unsqueeze(-1)).sum(-1)
            result = torch.einsum("tkhs,skd->tkhd", scores.softmax(-1), key[0].float()).flatten(1, 2).contiguous()
            return (
                result,
                maximum.permute(1, 0, 2).contiguous(),
                total.permute(1, 0, 2).contiguous(),
            )

        def native(**kwargs):
            packed_query = kwargs["query"]
            self.assertTrue(packed_query.is_contiguous())
            query, query_rope = packed_query[..., :4], packed_query[..., 4:]
            native_group = query.shape[1] // kv_heads
            self.assertEqual(native_group, group)
            self.assertIs(kwargs["key"], key)
            self.assertIs(kwargs["value"], key)
            self.assertIs(kwargs["block_table"], table)
            self.assertIs(kwargs["sparse_indices"], topk)
            self.assertEqual(kwargs["rope_head_dim"], rope_dim)
            return reference(query, query_rope)

        expected = reference(q, rope)
        packed_query = torch.cat((q, rope), dim=-1).transpose(0, 1).contiguous().transpose(0, 1) if packed else None
        runtime = adaptor._execute_kv_quant_sparse_flash_attention_vllm.__globals__["torch_npu"]
        with (
            patch.object(
                torch.ops._C_ascend, "npu_kv_quant_sparse_flash_attention_vllm", create=True, side_effect=native
            ),
            patch.object(
                runtime,
                "npu_kv_quant_sparse_flash_attention",
                create=True,
                side_effect=AssertionError("QSFA must use the custom operator"),
            ),
            patch.object(F, "pad", side_effect=AssertionError("QSFA must not pad query heads")),
            patch("torch.cat", wraps=torch.cat) as cat,
        ):
            actual = adaptor.execute_sparse_flash_attention_process(
                SimpleNamespace(scale=0.5, is_kimi_k31=True, qk_rope_head_dim=rope_dim),
                None if packed else q,
                None if packed else rope,
                (key,),
                topk,
                SimpleNamespace(block_table=table),
                torch.tensor([2]),
                torch.tensor([3]),
                return_lse=return_lse,
                packed_query=packed_query,
            )
        if packed:
            cat.assert_not_called()
        for output, want in zip(actual if return_lse else (actual,), expected):
            torch.testing.assert_close(output, want)
            self.assertTrue(output.is_contiguous())


if __name__ == "__main__":
    unittest.main()
