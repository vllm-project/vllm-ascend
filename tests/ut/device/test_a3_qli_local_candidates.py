# SPDX-License-Identifier: Apache-2.0
"""Exercise the A3 adaptor contract without importing torch or the NPU runtime.

The native boundary is replaced because this test also runs on development
hosts without PyTorch. Tensor doubles retain shape, dtype and page-view identity;
operator arithmetic is covered by the separate A3 hardware contract tests.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class Tensor:
    def __init__(self, shape, dtype, *, storage=None, stride0=None, data=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.storage = storage if storage is not None else object()
        self.stride0 = stride0
        self.data = data

    def to(self, dtype):
        if dtype == self.dtype:
            return self
        return Tensor(self.shape, dtype, data=self.data)

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        if shape[0] == -1:
            shape = (self.shape[0], *shape[1:])
        return Tensor(shape, self.dtype, storage=self.storage, data=self.data)

    def contiguous(self):
        return self

    def squeeze(self, dim):
        assert self.shape[dim] == 1
        return Tensor(
            self.shape[:dim] + self.shape[dim + 1 :],
            self.dtype,
            storage=self.storage,
            stride0=self.stride0,
            data=self.data,
        )


def load_adaptor(native, metadata, *, native_int8=True):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/device/device_op.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    adaptor = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "BaseDeviceAdaptor")
    adaptor.body = [
        n
        for n in adaptor.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"build_indexer_local_metadata", "indexer_select_local_candidates"}
    ]

    def pad(tensor, padding, value):
        raise AssertionError("Producer query_start_loc already contains the leading zero")

    namespace = {
        "get_current_hardware_profile": lambda: None,
        "uses_native_int8_sparse_adaptor": lambda profile: native_int8,
        "_load_a3_dcp_qli": lambda: None,
        "torch": SimpleNamespace(
            Tensor=Tensor,
            int8="int8",
            int32="int32",
            float16="float16",
            ops=SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_quant_lightning_indexer_v2_metadata=metadata,
                    npu_quant_lightning_indexer_dcp_metadata=metadata,
                    npu_quant_lightning_indexer_local_candidates=native,
                )
            ),
        ),
        "F": SimpleNamespace(pad=pad),
    }
    exec(compile(ast.Module(body=[adaptor], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["BaseDeviceAdaptor"]


class TestA3QLILocalCandidates(unittest.TestCase):
    def test_non_int8_inputs_do_not_enter_the_a3_int8_kernel(self):
        def unexpected(*args, **kwargs):
            self.fail("INT8 native QLI was called with a non-INT8 query or key")

        adaptor = load_adaptor(unexpected, unexpected)
        other = Tensor((1, 1, 128), "int8")
        for query_dtype, key_dtype in (("float8", "int8"), ("int8", "float8"), ("bfloat16", "bfloat16")):
            query = Tensor((1, 1, 128), query_dtype)
            key = Tensor((1, 1, 128), key_dtype)
            with self.assertRaises(NotImplementedError):
                adaptor.indexer_select_local_candidates(query, other, key, *([other] * 6), qli_metadata=object())

    def test_other_standard_devices_do_not_enter_the_a3_int8_kernel(self):
        def unexpected(*args, **kwargs):
            self.fail("A3 native QLI was called on another standard device")

        adaptor = load_adaptor(unexpected, unexpected, native_int8=False)
        tensor = Tensor((1, 1, 128), "int8")
        with self.assertRaises(NotImplementedError):
            adaptor.indexer_select_local_candidates(*([tensor] * 9), qli_metadata=object())
        with self.assertRaises(NotImplementedError):
            adaptor.build_indexer_local_metadata(
                tensor,
                tensor,
                num_heads_q=32,
                head_dim=128,
                topk=2048,
                causal_seqlens_kv_global=tensor,
                cp_world=16,
                cp_rank=7,
                cp_kv_cache_interleave_size=768,
            )

    def test_int8_query_uses_fp16_scales_and_preserves_independent_history_views(self):
        key = Tensor((7, 128, 1, 128), "int8", stride0=16640)
        scale = Tensor((7, 128, 1, 1), "float16", stride0=8320)
        q = Tensor((5, 32, 128), "int8")
        weights = Tensor((5, 32), "bfloat16")
        q_scale = Tensor((5, 32, 1), "float16")
        block_table = Tensor((2, 7), "int32")
        visible = Tensor((5,), "int64", data=[0, 1, 7, 128, 1024])
        output = (object(), object())
        metadata_buffer = object()
        query_start_loc = Tensor((3,), "int32", data=[0, 2, 5])
        seq_lens = Tensor((2,), "int32", data=[0, 1024])
        global_seq_lens = Tensor((2,), "int32", data=[1, 20000])
        metadata_calls = []

        def metadata(*args, **kwargs):
            metadata_calls.append((args, kwargs))
            self.assertEqual(args, (32, 1, 128, 2048, 2))
            self.assertEqual(kwargs["cu_seqlens_q"].data, [0, 2, 5])
            self.assertIs(kwargs["cu_seqlens_q"], query_start_loc)
            self.assertIs(kwargs["seqused_k"], seq_lens)
            self.assertEqual(kwargs["seqused_k"].dtype, "int32")
            self.assertEqual(kwargs["batch_size"], 2)
            self.assertEqual(kwargs["layout_q"], "TND")
            self.assertEqual(kwargs["layout_k"], "PA_BBND")
            self.assertEqual(kwargs["mask_mode"], 3)
            self.assertEqual(kwargs["enable_dcp"], 1)
            self.assertIs(kwargs["causal_seqlens_kv_global"], global_seq_lens)
            self.assertEqual(kwargs["cp_world"], 16)
            self.assertEqual(kwargs["cp_rank"], 7)
            self.assertEqual(kwargs["cp_kv_cache_interleave_size"], 768)
            self.assertEqual(kwargs["cmp_ratio"], 1)
            return metadata_buffer

        def native(*args):
            self.assertIs(args[0], q)
            self.assertIs(args[1], key)
            self.assertEqual(args[1].stride0, 16640)
            self.assertEqual(args[2].dtype, "float16")
            self.assertEqual(args[3].shape, (5, 32))
            self.assertEqual(args[3].dtype, "float16")
            self.assertIs(args[4].storage, scale.storage)
            self.assertEqual(args[4].stride0, 8320)
            self.assertEqual(args[4].shape, (7, 128, 1))
            self.assertEqual(args[5].data, [0, 2, 5])
            self.assertIs(args[5], query_start_loc)
            self.assertIs(args[6], seq_lens)
            self.assertIs(args[7], block_table)
            self.assertEqual(args[8].dtype, "int32")
            self.assertEqual(args[8].shape, (5, 1))
            self.assertEqual(args[8].data, [0, 1, 7, 128, 1024])
            self.assertIs(args[9], metadata_buffer)
            self.assertEqual(args[10], 2048)
            return output

        try:
            adaptor = load_adaptor(native, metadata)
            plan = adaptor.build_indexer_local_metadata(
                query_start_loc,
                seq_lens,
                num_heads_q=32,
                head_dim=128,
                topk=2048,
                causal_seqlens_kv_global=global_seq_lens,
                cp_world=16,
                cp_rank=7,
                cp_kv_cache_interleave_size=768,
            )
            self.assertIs(plan, metadata_buffer)
            for _ in range(2):
                result = adaptor.indexer_select_local_candidates(
                    q,
                    weights,
                    key,
                    scale,
                    block_table,
                    query_start_loc,
                    seq_lens,
                    q_scale,
                    visible,
                    qli_metadata=plan,
                )
                self.assertIs(result, output)
        except NotImplementedError as error:
            self.fail(str(error))
        self.assertEqual(len(metadata_calls), 1)


if __name__ == "__main__":
    unittest.main()
