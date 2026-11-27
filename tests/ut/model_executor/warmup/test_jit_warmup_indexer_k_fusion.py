# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration and compile-only launch coverage; no kernel math runs here."""

import inspect
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from tests.ut.model_executor.warmup.helpers import make_registry
from tests.ut.ops.test_indexer_k_fusion import _load_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor


class TestIndexerKFusionWarmup(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_launcher("vllm_ascend/ops/triton/indexer_k_fusion.py")

    def setUp(self):
        self.owner = self.module.IndexerKFusionKernel()
        self.config = dict(head_dim=128, rotary_dim=64)

    def test_keys_cover_pointer_dtypes_and_keep_vector_layout(self):
        keys = self.owner.get_warmup_keys(**self.config)
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(set(keys)), 4)
        self.assertEqual(
            {(key.weight_dtype, key.slots_dtype) for key in keys},
            {(weight, slots) for weight in (torch.bfloat16, torch.float32) for slots in (torch.int32, torch.int64)},
        )
        strided = self.owner.get_warmup_keys(**self.config, vector_strides=(2, 2, 2))
        self.assertTrue(set(keys).isdisjoint(strided))
        for key in keys + strided:
            inputs = self.owner.warmup_inputs(key)
            self.assertEqual(inputs["weight"].stride(0), key.weight_stride)
            self.assertEqual(inputs["cos"].stride(1), key.cos_col_stride)
            self.assertEqual(inputs["sin"].stride(1), key.sin_col_stride)
            self.assertEqual(self.owner.compile_key(vars(key)), key)

    def test_compile_binds_every_argument_without_executing_kernel(self):
        raw_kernel = type(self.owner).kernel
        kernel = MagicMock(arg_names=raw_kernel.arg_names)
        keys = self.owner.get_warmup_keys(**self.config)
        with patch.object(self.owner, "kernel", kernel):
            for key in keys:
                self.owner.compile(key)
                kwargs = dict(kernel.warmup.call_args.kwargs)
                self.assertEqual(kwargs.pop("grid"), (1,))
                self.assertIs(kwargs.pop("enable_fp_fusion"), False)
                bound = inspect.signature(raw_kernel).bind(**kwargs).arguments
                self.assertEqual(bound["HEAD_DIM"], key.head_dim)
                self.assertEqual(bound["ROTARY_DIM"], key.rotary_dim)
                self.assertEqual(bound["WEIGHT_STRIDE"], key.weight_stride)
                self.assertEqual(bound["slots_ptr"].dtype, key.slots_dtype)
                self.assertEqual(bound["weight_ptr"].dtype, key.weight_dtype)
                self.assertIsInstance(bound["key_cache_ptr"], TritonWarmupTensor)
                self.assertIsInstance(bound["rms_eps"], float)
        self.assertEqual(kernel.warmup.call_count, len(keys))
        kernel.__getitem__.assert_not_called()
        self.assertFalse(self.owner._warming)

    def test_registration_window_and_duplicate_key_compilation(self):
        registry = make_registry()
        self.owner.register_warmup(**self.config)
        self.assertEqual(len(registry), 0)
        with registry.activate():
            self.owner.register_warmup(**self.config)
            self.owner.register_warmup(**self.config)
        self.assertEqual(len(registry), 1)
        with (
            patch.object(self.owner, "compile") as compile_key,
            patch("vllm.distributed.is_global_first_rank", return_value=False),
        ):
            registry.warmup()
        self.assertEqual(compile_key.call_count, 4)

    def test_runtime_grid_and_physical_strides_use_same_launcher(self):
        inputs = self.owner.warmup_inputs(self.owner.get_warmup_keys(**self.config)[0])
        inputs.update(
            input=TritonWarmupTensor(torch.bfloat16, shape=(17, 128), strides=(144, 1)),
            key_cache=TritonWarmupTensor(torch.int8, shape=(8, 32, 1, 128), strides=(5000, 144, 128, 1)),
            scale_cache=TritonWarmupTensor(torch.float16, shape=(8, 32, 1, 1), strides=(80, 2, 1, 1)),
        )
        kernel = MagicMock(arg_names=self.owner.kernel.arg_names)
        with patch.object(self.owner, "kernel", kernel):
            self.owner(**inputs)
        kernel.__getitem__.assert_called_once_with((17,))
        kwargs = kernel.__getitem__.return_value.call_args.kwargs
        self.assertEqual(kwargs["input_row_stride"], 144)
        self.assertEqual(kwargs["key_block_stride"], 5000)
        self.assertEqual(kwargs["scale_row_stride"], 2)
        self.assertEqual(kwargs["num_blocks"], 8)
        self.assertEqual(kwargs["page_size"], 32)
        kernel.warmup.assert_not_called()
