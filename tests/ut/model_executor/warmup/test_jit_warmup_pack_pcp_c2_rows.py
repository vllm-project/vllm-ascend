# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packing warmup contracts; compiler-boundary stubs never execute NPU math."""

import importlib.util
import inspect
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor

from tests.ut.model_executor.warmup.helpers import make_registry


def _load_packing_module():
    def jit(fn=None, **kwargs):
        def decorate(kernel):
            kernel.arg_names = tuple(inspect.signature(kernel).parameters)
            return kernel

        return decorate(fn) if fn is not None else decorate

    utils = ModuleType("vllm.triton_utils")
    utils.HAS_TRITON = False
    utils.tl = SimpleNamespace(constexpr=object())
    utils.triton = SimpleNamespace(
        jit=jit,
        autotune=lambda **kwargs: lambda fn: fn,
        Config=MagicMock(),
        cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
    )
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/triton/compressor/compressor_triton.py"
    spec = importlib.util.spec_from_file_location("_pack_pcp_c2_warmup_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"vllm.triton_utils": utils, spec.name: module}):
        spec.loader.exec_module(module)
    return module


class TestPackPCPC2RowsWarmup(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_packing_module()

    def setUp(self):
        self.owner = self.module.PackPCPC2RowsKernel()

    def test_production_key_has_model_width_and_contiguous_strides(self):
        for width in (1, 127, 128, 129, 512):
            keys = self.owner.get_warmup_keys(width=width)
            self.assertEqual(len(keys), 1)
            key = keys[0]
            self.assertEqual(key.width, width)
            self.assertEqual((key.kv_row_stride, key.kv_col_stride), (width, 1))
            self.assertEqual((key.score_row_stride, key.score_col_stride), (width, 1))
            self.assertEqual(self.owner.compile_key(vars(key)), key)
            inputs = self.owner.warmup_inputs(key)
            self.assertEqual(inputs["kv"].dtype, torch.float32)
            self.assertEqual(inputs["scores"].dtype, torch.float32)
            self.assertEqual(inputs["send_indices"].dtype, torch.int64)
            self.assertEqual(inputs["valid_mask"].dtype, torch.bool)
            self.assertEqual(inputs["packed_output"].dtype, torch.float32)

    def test_registering_another_layout_or_width_adds_distinct_keys(self):
        keys = [
            self.owner.get_warmup_keys(width=512)[0],
            self.owner.get_warmup_keys(width=129)[0],
            self.owner.get_warmup_keys(width=512, projection_strides=(1024, 2, 1, 17))[0],
            self.owner.get_warmup_keys(width=512, projection_strides=(0, 1, 512, 1))[0],
        ]
        self.assertEqual(len(set(keys)), 4)
        for key in keys:
            inputs = self.owner.warmup_inputs(key)
            self.assertEqual(inputs["kv"].stride(), (key.kv_row_stride, key.kv_col_stride))
            self.assertEqual(inputs["scores"].stride(), (key.score_row_stride, key.score_col_stride))

    def test_compile_binds_all_kernel_arguments_and_does_not_launch(self):
        raw = type(self.owner).kernel
        kernel = MagicMock(arg_names=raw.arg_names)
        for width in (1, 129, 512):
            key = self.owner.get_warmup_keys(width=width)[0]
            with patch.object(self.owner, "kernel", kernel):
                self.owner.compile(key)
            kwargs = dict(kernel.warmup.call_args.kwargs)
            self.assertEqual(kwargs.pop("grid"), (1,))
            bound = inspect.signature(raw).bind(**kwargs).arguments
            self.assertEqual(bound["WIDTH"], width)
            self.assertEqual(bound["BLOCK_SIZE"], 128)
            self.assertEqual(bound["num_tokens"], 1)
            self.assertEqual(bound["KV_ROW_STRIDE"], width)
            self.assertEqual(bound["SCORE_COL_STRIDE"], 1)
            self.assertIsInstance(bound["packed_output_ptr"], TritonWarmupTensor)
        self.assertEqual(kernel.warmup.call_count, 3)
        kernel.__getitem__.assert_not_called()
        self.assertFalse(self.owner._warming)

    def test_registration_window_and_compile_key_deduplication(self):
        registry = make_registry()
        self.owner.register_warmup(width=512)
        self.assertEqual(len(registry), 0)
        with registry.activate():
            self.owner.register_warmup(width=512)
            self.owner.register_warmup(width=512)
            self.owner.register_warmup(width=129)
        with (
            patch.object(self.owner, "compile") as compile_key,
            patch("vllm.distributed.is_global_first_rank", return_value=False),
        ):
            registry.warmup()
        self.assertEqual(compile_key.call_count, 2)
        self.assertEqual({call.args[0].width for call in compile_key.call_args_list}, {129, 512})

    def test_runtime_grid_uses_actual_rows_and_column_tiles(self):
        for tokens, rows, width in ((0, 5, 129), (1, 1, 1), (17, 23, 512), (16, 7, 128)):
            key = self.owner.get_warmup_keys(width=width)[0]
            inputs = self.owner.warmup_inputs(key)
            inputs.update(
                kv=TritonWarmupTensor(torch.float32, shape=(tokens, width)),
                scores=TritonWarmupTensor(torch.float32, shape=(tokens, width)),
                send_indices=TritonWarmupTensor(torch.int64, shape=(rows,)),
                valid_mask=TritonWarmupTensor(torch.bool, shape=(rows,)),
                packed_output=TritonWarmupTensor(torch.float32, shape=(rows, 2 * width)),
            )
            kernel = MagicMock(arg_names=self.owner.kernel.arg_names)
            with patch.object(self.owner, "kernel", kernel):
                self.owner(**inputs)
            kernel.__getitem__.assert_called_once_with((rows, (width + 127) // 128))
            kwargs = kernel.__getitem__.return_value.call_args.kwargs
            self.assertEqual(kwargs["num_tokens"], tokens)
            self.assertIs(kwargs["valid_mask_ptr"], inputs["valid_mask"])
            self.assertIs(kwargs["packed_output_ptr"], inputs["packed_output"])
            kernel.warmup.assert_not_called()

    @staticmethod
    def _tensor_metadata(shape, dtype, storage, strides=None):
        device = SimpleNamespace(type="npu", index=0)
        tensor = MagicMock(shape=shape, ndim=len(shape), dtype=dtype, device=device, layout=torch.strided)
        tensor.numel.return_value = math.prod(shape)
        tensor.is_contiguous.return_value = True
        tensor.untyped_storage.return_value.data_ptr.return_value = storage
        if strides is not None:
            tensor.stride.side_effect = lambda dim: strides[dim]
        return tensor

    def _public_inputs(self, tokens, rows, width, kv_stride, score_stride):
        return dict(
            kv=self._tensor_metadata((tokens, width), torch.float32, 1, kv_stride),
            scores=self._tensor_metadata((tokens, width), torch.float32, 2, score_stride),
            send_indices=self._tensor_metadata((rows,), torch.int64, 3),
            valid_mask=self._tensor_metadata((rows,), torch.bool, 4),
            packed_output=self._tensor_metadata((rows, 2 * width), torch.float32, 5),
        )

    def test_public_entry_keeps_strided_views_and_caller_output(self):
        owner = self.module._PACK_PCP_C2_ROWS_KERNEL
        for kv_stride, score_stride in (((129, 1), (129, 1)), ((258, 2), (1, 17)), ((0, 1), (145, 1))):
            inputs = self._public_inputs(17, 23, 129, kv_stride, score_stride)
            kernel = MagicMock(arg_names=owner.kernel.arg_names)
            with patch.object(owner, "kernel", kernel):
                actual = self.module.pack_pcp_c2_rows(**inputs)
            self.assertIs(actual, inputs["packed_output"])
            kwargs = kernel.__getitem__.return_value.call_args.kwargs
            self.assertIs(kwargs["packed_output_ptr"], actual)
            self.assertEqual((kwargs["KV_ROW_STRIDE"], kwargs["KV_COL_STRIDE"]), kv_stride)
            self.assertEqual((kwargs["SCORE_ROW_STRIDE"], kwargs["SCORE_COL_STRIDE"]), score_stride)

    def test_public_empty_output_returns_without_launch_or_storage_access(self):
        inputs = self._public_inputs(8, 0, 64, (64, 1), (64, 1))
        with patch.object(self.module, "_PACK_PCP_C2_ROWS_KERNEL") as owner:
            result = self.module.pack_pcp_c2_rows(**inputs)
        self.assertIs(result, inputs["packed_output"])
        owner.assert_not_called()
        for value in inputs.values():
            value.untyped_storage.assert_not_called()

    def test_public_empty_projection_still_launches_zero_fill_for_send_rows(self):
        inputs = self._public_inputs(0, 5, 129, (129, 1), (129, 1))
        owner = self.module._PACK_PCP_C2_ROWS_KERNEL
        kernel = MagicMock(arg_names=owner.kernel.arg_names)
        with patch.object(owner, "kernel", kernel):
            result = self.module.pack_pcp_c2_rows(**inputs)
        self.assertIs(result, inputs["packed_output"])
        kernel.__getitem__.assert_called_once_with((5, 2))
        self.assertEqual(kernel.__getitem__.return_value.call_args.kwargs["num_tokens"], 0)

    def test_public_alias_rejected_before_wrapper_launch(self):
        inputs = self._public_inputs(17, 23, 129, (129, 1), (129, 1))
        inputs["packed_output"].untyped_storage.return_value.data_ptr.return_value = 1
        with patch.object(self.module, "_PACK_PCP_C2_ROWS_KERNEL") as owner:
            with self.assertRaisesRegex(ValueError, "share storage"):
                self.module.pack_pcp_c2_rows(**inputs)
        owner.assert_not_called()
