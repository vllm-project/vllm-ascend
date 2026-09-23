#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Input customization for QuantLightningIndexer V2 TTK cases."""

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

QUANT_MODE_MXFP4 = 5
QUANT_MODE_MXFP8 = 3
QUANT_MODE_HIF4 = 6


def restore_mx_input_dtypes(query, key, query_scale, key_scale, quant_mode):
    """Restore packed MX dtypes needed by the existing replay compare path."""
    quant_mode = int(quant_mode)
    if quant_mode == QUANT_MODE_MXFP8:
        qk_dtype = getattr(torch, "float8_e4m3fn", None)
    elif quant_mode == QUANT_MODE_MXFP4:
        qk_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    else:
        return query, key, query_scale, key_scale
    scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if qk_dtype is None or scale_dtype is None:
        raise RuntimeError("current PyTorch does not provide the requested MX dtype")
    if query.dtype == torch.uint8:
        query = query.view(qk_dtype)
    if key.dtype == torch.uint8:
        key = key.view(qk_dtype)
    if query_scale.dtype == torch.uint8:
        query_scale = query_scale.view(scale_dtype)
    if key_scale.dtype == torch.uint8:
        key_scale = key_scale.view(scale_dtype)
    return query, key, query_scale, key_scale


class QuantLightningIndexerV2InputAdapter:
    """Translate a TTK case and reuse the pytest input/golden generator."""

    @staticmethod
    def module_load_error(stage, path, exc):
        return RuntimeError(
            "Failed to load QuantLightningIndexerV2 module; "
            f"stage={stage}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def __init__(self):
        self.pytest_golden = None
        self.pytest_normalizer = None
        self.batch_consistency = None

    def load_batch_consistency(self):
        if self.batch_consistency is not None:
            return self.batch_consistency
        name = "qli_v2_ttk_batch_consistency"
        path = Path(__file__).with_name("batch_consistency.py")
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.batch_consistency = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error("assets batch consistency", path, exc) from exc

    @staticmethod
    def load_golden_store():
        name = "qli_v2_ttk_golden"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("golden.py")
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise QuantLightningIndexerV2InputAdapter.module_load_error("assets Golden store", path, exc) from exc
        return module

    @staticmethod
    def load_metadata_protocol():
        name = "qli_v2_ttk_metadata_protocol"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("metadata_protocol.py")
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise QuantLightningIndexerV2InputAdapter.module_load_error("assets metadata protocol", path, exc) from exc
        return module

    def load_pytest_golden(self):
        if self.pytest_golden is not None:
            return self.pytest_golden
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "quant_lightning_indexer_v2_golden.py"
        name = "qli_v2_pytest_golden"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.pytest_golden = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error("pytest Golden", path, exc) from exc
        finally:
            if inserted:
                sys.path.remove(str(pytest_dir))

    def load_pytest_normalizer(self):
        if self.pytest_normalizer is not None:
            return self.pytest_normalizer
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "qliv2_parameter_normalization.py"
        name = "qli_v2_pytest_normalizer"
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self.pytest_normalizer = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error("pytest parameter normalizer", path, exc) from exc

    @staticmethod
    def list_value(kwargs, name):
        value = kwargs.get(f"{name}_values")
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, np.ndarray):
            value = value.reshape(-1).tolist()
        return [int(item) for item in value]

    @staticmethod
    def tensor_dtype(tensor):
        if torch.is_tensor(tensor):
            return tensor.dtype
        if "hifloat8" in str(tensor.dtype):
            return torch.uint8
        return torch.from_numpy(np.asarray(tensor)).dtype

    @staticmethod
    def to_cpu_tensor(tensor):
        if tensor is None:
            return None
        if torch.is_tensor(tensor):
            return tensor.detach().cpu()
        array = np.asarray(tensor)
        if "hifloat8" in str(array.dtype):
            array = array.view(np.uint8)
        return torch.from_numpy(np.array(array, copy=True))

    @staticmethod
    def prefix_lengths(value):
        if not value:
            return []
        return [int(value[index + 1]) - int(value[index]) for index in range(len(value) - 1)]

    @staticmethod
    def data_range(input_ranges, index):
        if input_ranges and index < len(input_ranges) and input_ranges[index] is not None:
            return repr(list(input_ranges[index]))
        return None

    @staticmethod
    def qk_dtype_name(tensor, quant_mode):
        if quant_mode == QUANT_MODE_MXFP8:
            return "FLOAT8_E4M3FN"
        if quant_mode == QUANT_MODE_MXFP4:
            return "FLOAT4_E2M1FN_X2"
        if quant_mode == QUANT_MODE_HIF4:
            return "HIF4"
        dtype = QuantLightningIndexerV2InputAdapter.tensor_dtype(tensor)
        dtype_name = str(tensor.dtype)
        if dtype == torch.int8:
            return "INT8"
        if "float8_e4m3fn" in dtype_name:
            return "FLOAT8_E4M3FN"
        if dtype == torch.uint8:
            return "HIFLOAT8"
        raise ValueError(f"unsupported QLI_V2 q/k dtype: {tensor.dtype}")

    @staticmethod
    def dequant_dtype_name(tensor, quant_mode):
        if quant_mode in (QUANT_MODE_MXFP8, QUANT_MODE_MXFP4):
            return "FLOAT8_E8M0FNU"
        dtype = QuantLightningIndexerV2InputAdapter.tensor_dtype(tensor)
        if dtype == torch.float16:
            return "FP16"
        if dtype == torch.float32:
            return "FP32"
        raise ValueError(f"unsupported QLI_V2 dequant dtype: {tensor.dtype}")

    @staticmethod
    def weight_dtype_name(tensor):
        mapping = {
            torch.int8: "INT8",
            torch.uint8: "UINT8",
            torch.float16: "FP16",
            torch.float32: "FP32",
            torch.bfloat16: "BF16",
        }
        dtype = QuantLightningIndexerV2InputAdapter.tensor_dtype(tensor)
        if dtype in mapping:
            return mapping[dtype]
        if "float8_e4m3fn" in str(tensor.dtype):
            return "FLOAT8_E4M3FN"
        raise ValueError(f"unsupported QLI_V2 weight dtype: {tensor.dtype}")

    @staticmethod
    def pytest_uses_weight_dtype(pytest_golden):
        parameters = inspect.signature(pytest_golden.GeneralizedQLIV2.__init__).parameters
        return "weight_dtype" in parameters

    def geometry(self, query, key, layout_query, layout_key, cu_q, cu_k, seq_q, seq_k, quant_mode):
        q_lengths = self.prefix_lengths(cu_q) or (seq_q or [])
        k_lengths = self.prefix_lengths(cu_k) or (seq_k or [])
        if layout_query == "BSND":
            batch_size, q_seq, q_head_num, head_dim = [int(item) for item in query.shape]
            q_t_size = 0
        elif layout_query == "TND":
            q_t_size, q_head_num, head_dim = [int(item) for item in query.shape]
            batch_size = len(q_lengths)
            q_seq = max(q_lengths, default=q_t_size)
        else:
            raise ValueError(f"unsupported QLI_V2 query layout: {layout_query}")

        is_aclnn_float4 = isinstance(query, np.ndarray) and "float4" in str(query.dtype)
        if quant_mode in (QUANT_MODE_MXFP4, QUANT_MODE_HIF4) and not is_aclnn_float4:
            head_dim *= 2

        if layout_key == "BSND":
            _, k_seq, k_head_num, _ = [int(item) for item in key.shape]
            k_t_size = 0
            block_size = 0
            block_num = 0
        elif layout_key == "TND":
            k_t_size, k_head_num, _ = [int(item) for item in key.shape]
            k_seq = max(k_lengths, default=k_t_size)
            block_size = 0
            block_num = 0
        elif layout_key == "PA_BBND":
            block_num, block_size, k_head_num, _ = [int(item) for item in key.shape]
            capacity = block_num * block_size
            per_batch_capacity = capacity // batch_size if batch_size > 0 else block_size
            k_seq = max(k_lengths, default=per_batch_capacity)
            k_t_size = 0
        else:
            raise ValueError(f"unsupported QLI_V2 key layout: {layout_key}")
        return (
            batch_size,
            q_seq,
            k_seq,
            q_t_size,
            k_t_size,
            q_head_num,
            k_head_num,
            head_dim,
            block_size,
            block_num,
        )

    def build_case_params(
        self,
        query,
        key,
        weights,
        query_dequant_scale,
        layout_query,
        layout_key,
        kwargs,
        include_weight_dtype=False,
    ):
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_k = self.list_value(kwargs, "cu_seqlens_k")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_k = self.list_value(kwargs, "seqused_k")
        residual = self.list_value(kwargs, "cmp_residual_k")
        quant_mode = kwargs.get("quant_mode")
        quant_mode = None if quant_mode is None else int(quant_mode)
        geometry = self.geometry(
            query,
            key,
            layout_query,
            layout_key,
            cu_q,
            cu_k,
            seq_q,
            seq_k,
            quant_mode,
        )
        input_ranges = kwargs.get("qli_input_ranges") or kwargs.get("input_ranges") or ()
        output_range = kwargs.get("output_idx_offset_range")
        output_range = None if output_range is None else repr(list(output_range))
        max_seqlen_q = kwargs.get("max_seqlen_q")
        max_seqlen_q = None if max_seqlen_q is None else int(max_seqlen_q)
        dtype_params = (self.qk_dtype_name(query, quant_mode),)
        if include_weight_dtype:
            dtype_params += (self.weight_dtype_name(weights),)
        dtype_params += (
            self.dequant_dtype_name(query_dequant_scale, quant_mode),
            "INT32",
        )
        params = (
            geometry
            + dtype_params
            + (
                cu_q,
                cu_k,
                seq_q,
                seq_k,
                residual,
                max_seqlen_q,
                quant_mode,
                layout_query,
                layout_key,
                kwargs.get("sparse_count"),
                kwargs.get("sparse_mode"),
                self.data_range(input_ranges, 0),
                self.data_range(input_ranges, 1),
                self.data_range(input_ranges, 2),
                self.data_range(input_ranges, 3),
                self.data_range(input_ranges, 4),
                kwargs.get("cmp_ratio"),
                kwargs.get("return_value"),
                output_range,
            )
        )
        return self.load_pytest_normalizer().normalize_qliv2_params(params)

    @staticmethod
    def copy_tensor(dst, src, name, packed_dtype=None):
        if dst is None:
            if src is not None:
                import logging

                logging.warning("%s is absent from CSV but pytest generator produced a tensor. Skipping copy.", name)
            return
        if src is None:
            raise ValueError(f"{name} is present in CSV but pytest generator returned None")
        src_cpu = QuantLightningIndexerV2InputAdapter.to_cpu_tensor(src)
        src_cpu = src_cpu.contiguous()
        if packed_dtype is not None:
            src_cpu = src_cpu.to(packed_dtype)
        if torch.is_tensor(dst) and dst.dtype == torch.uint8 and src_cpu.element_size() == 1:
            src_cpu = src_cpu.view(torch.uint8)
        if tuple(dst.shape) != tuple(src_cpu.shape):
            raise ValueError(f"{name} shape mismatch: TTK={tuple(dst.shape)} pytest={tuple(src_cpu.shape)}")
        if torch.is_tensor(dst):
            src_tensor = torch.as_tensor(src_cpu)
            dst.copy_(src_tensor.to(dtype=dst.dtype, device=dst.device))
            return

        dst_array = np.asarray(dst)
        if "float8" in str(src_cpu.dtype):
            # ACLNN keeps this storage as a NumPy custom dtype.  Copy its
            # encoded bytes directly so assets do not depend on TTK's dtype helpers.
            src_array = src_cpu.view(torch.uint8).numpy()
            np.copyto(dst_array.view(np.uint8), src_array)
            return
        else:
            src_array = np.asarray(src_cpu)
        if "hifloat8" in str(dst_array.dtype):
            np.copyto(dst_array.view(np.uint8), src_array.view(np.uint8))
        else:
            np.copyto(dst_array, src_array.astype(dst_array.dtype, copy=False))

    @staticmethod
    def tensor_values(tensor):
        if tensor is None:
            return None
        tensor = QuantLightningIndexerV2InputAdapter.to_cpu_tensor(tensor)
        return [int(value) for value in tensor.reshape(-1).tolist()]

    @staticmethod
    def unpack_mxfp4(tensor, fp4_values):
        """Unpack two FP4 E2M1 values stored in each uint8 byte."""
        packed = tensor.view(torch.uint8).contiguous()
        codes = torch.stack(
            (packed & 0x0F, packed >> 4),
            dim=-1,
        ).flatten(-2)
        return fp4_values[codes.to(torch.long)]

    @staticmethod
    def restore_paged_tensor(tensor, block_table, batch_size, sequence_length):
        """Restore a paged key or scale tensor for the pytest compare model."""
        physical = QuantLightningIndexerV2InputAdapter.to_cpu_tensor(tensor)
        table = QuantLightningIndexerV2InputAdapter.to_cpu_tensor(block_table).to(torch.int64)
        if physical.ndim < 3:
            raise ValueError(f"paged tensor must have at least 3 dimensions, got shape {tuple(physical.shape)}")
        block_size, head_num = int(physical.shape[1]), int(physical.shape[2])
        trailing = tuple(int(dim) for dim in physical.shape[3:])
        logical = torch.zeros(
            (batch_size, head_num, sequence_length, *trailing),
            dtype=physical.dtype,
        )
        for batch_idx in range(batch_size):
            for logical_block, block_id_value in enumerate(table[batch_idx].tolist()):
                if block_id_value < 0:
                    continue
                if block_id_value >= physical.shape[0]:
                    raise ValueError(f"block id {block_id_value} exceeds paged block count")
                start = logical_block * block_size
                if start >= sequence_length:
                    break
                count = min(block_size, sequence_length - start)
                block = physical[block_id_value, :count]
                permutation = (1, 0, *range(2, block.ndim))
                logical[batch_idx, :, start : start + count] = block.permute(*permutation)
        return logical

    @staticmethod
    def normalize_compare_attributes(compare_context):
        attributes = dict(compare_context.attributes)
        aliases = {
            "topk": "sparse_count",
            "mask_mode": "sparse_mode",
            "layout_q": "layout_query",
            "layout_k": "layout_key",
            "quantMode": "quant_mode",
            "maxSeqlenQ": "max_seqlen_q",
            "layoutQOptional": "layout_query",
            "layoutKOptional": "layout_key",
            "maskMode": "sparse_mode",
            "cmpRatio": "cmp_ratio",
            "returnValue": "return_value",
        }
        for source, target in aliases.items():
            if target not in attributes and source in attributes:
                attributes[target] = attributes[source]
        return attributes

    def rebuild_compare_data(self, compare_context):
        """Rebuild only the pytest TopK compare context from replayed inputs."""
        tensors = tuple(compare_context.input_tensors or ())
        if len(tensors) < 12:
            raise ValueError("QuantLightningIndexerV2 compare context requires twelve input slots")
        (
            query,
            key,
            weights,
            query_scale,
            key_scale,
            cu_q,
            cu_k,
            seq_q,
            seq_k,
            residual,
            block_table,
            offset,
        ) = tensors[:12]
        attributes = self.normalize_compare_attributes(compare_context)
        for name, tensor in (
            ("cu_seqlens_q", cu_q),
            ("cu_seqlens_k", cu_k),
            ("seqused_q", seq_q),
            ("seqused_k", seq_k),
            ("cmp_residual_k", residual),
        ):
            values = self.tensor_values(tensor)
            if values is not None:
                attributes[f"{name}_values"] = values

        layout_q = attributes.get("layout_query")
        layout_k = attributes.get("layout_key")
        if layout_q is None or layout_k is None:
            raise ValueError("QLI_V2 replay compare requires layout_query and layout_key from attributes")
        pytest_golden = self.load_pytest_golden()
        uses_weight_dtype = self.pytest_uses_weight_dtype(pytest_golden)
        params = self.build_case_params(
            query,
            key,
            weights,
            query_scale,
            layout_q,
            layout_k,
            attributes,
            include_weight_dtype=uses_weight_dtype,
        )
        if uses_weight_dtype:
            (
                batch_size,
                q_seq,
                k_seq,
                q_t_size,
                k_t_size,
                q_head_num,
                k_head_num,
                head_dim,
                block_size,
                block_num,
                _,
                _,
                _,
                _,
                cu_q_values,
                cu_k_values,
                seq_q_values,
                seq_k_values,
                residual_values,
                max_seqlen_q,
                quant_mode,
                _,
                _,
                sparse_count,
                sparse_mode,
                _,
                _,
                _,
                _,
                _,
                cmp_ratio,
                return_value,
                _,
            ) = params
        else:
            (
                batch_size,
                q_seq,
                k_seq,
                q_t_size,
                k_t_size,
                q_head_num,
                k_head_num,
                head_dim,
                block_size,
                block_num,
                _,
                _,
                _,
                cu_q_values,
                cu_k_values,
                seq_q_values,
                seq_k_values,
                residual_values,
                max_seqlen_q,
                quant_mode,
                _,
                _,
                sparse_count,
                sparse_mode,
                _,
                _,
                _,
                _,
                _,
                cmp_ratio,
                return_value,
                _,
            ) = params

        q_lengths = self.prefix_lengths(cu_q_values) if layout_q == "TND" else (seq_q_values or [q_seq] * batch_size)
        k_lengths = self.prefix_lengths(cu_k_values) if layout_k == "TND" else (seq_k_values or [k_seq] * batch_size)
        residual_for_cpu = [0] * batch_size if cmp_ratio == 1 or sparse_mode == 0 else list(residual_values)

        def as_int_tensor(value):
            return None if value is None else torch.tensor(value, dtype=torch.int32)

        cu_q_cpu = as_int_tensor(cu_q_values)
        cu_k_cpu = as_int_tensor(cu_k_values)
        seq_q_cpu = as_int_tensor(seq_q_values)
        seq_k_cpu = as_int_tensor(seq_k_values)
        query, key, query_scale, key_scale = restore_mx_input_dtypes(
            self.to_cpu_tensor(query),
            self.to_cpu_tensor(key),
            self.to_cpu_tensor(query_scale),
            self.to_cpu_tensor(key_scale),
            quant_mode,
        )
        weights = self.to_cpu_tensor(weights)
        block_table = self.to_cpu_tensor(block_table)
        offset = self.to_cpu_tensor(offset)
        qk_dtype = query.dtype
        if quant_mode == QUANT_MODE_MXFP4:
            query = self.unpack_mxfp4(query, pytest_golden.FP4_E2M1_VALUES)
            key = self.unpack_mxfp4(key, pytest_golden.FP4_E2M1_VALUES)
        model_args = [
            batch_size,
            q_seq,
            k_seq,
            q_t_size,
            k_t_size,
            q_head_num,
            k_head_num,
            head_dim,
            block_size,
            block_num,
            qk_dtype,
        ]
        if uses_weight_dtype:
            model_args.append(weights.dtype)
        model_args.extend(
            [
                query_scale.dtype,
                torch.int32,
                cu_q_cpu,
                cu_k_cpu,
                q_lengths,
                k_lengths,
                residual_for_cpu,
                max_seqlen_q,
                quant_mode,
                layout_q,
                layout_k,
                sparse_count,
                sparse_mode,
                None,
                None,
                None,
                None,
                None,
                cmp_ratio,
                return_value,
            ]
        )
        model = pytest_golden.GeneralizedQLIV2(*model_args)

        key_for_cpu = key
        key_scale_for_cpu = key_scale
        if layout_k == "PA_BBND":
            if block_table is None or not k_lengths:
                raise ValueError("PA_BBND compare context requires block_table and seqused_k")
            sequence_length = max(k_lengths)
            key_for_cpu = self.restore_paged_tensor(key, block_table, batch_size, sequence_length)
            if quant_mode != 4:
                key_scale_for_cpu = self.restore_paged_tensor(key_scale, block_table, batch_size, sequence_length)

        query_scale_for_cpu = query_scale
        if quant_mode == 4:
            query_scale_for_cpu = torch.full(
                tuple(query.shape[:-1]),
                query_scale.reshape(-1)[0].item(),
                dtype=query_scale.dtype,
            )
            if layout_k == "PA_BBND":
                key_scale_shape = (batch_size, k_head_num, max(k_lengths))
            else:
                key_scale_shape = tuple(key.shape[:-1])
            key_scale_for_cpu = torch.full(
                key_scale_shape,
                key_scale.reshape(-1)[0].item(),
                dtype=key_scale.dtype,
            )

        _, scores, _ = model.forward(
            query,
            key_for_cpu,
            weights,
            query_scale_for_cpu,
            key_scale_for_cpu,
            cu_q_cpu,
            cu_k_cpu,
            seq_q_cpu,
            seq_k_cpu,
            block_table,
            offset,
        )
        return {
            "params": params,
            "scores": scores,
            "topk_value": scores,
            "output_idx_offset": None if offset is None else offset.detach().cpu(),
            "score_layout": layout_q,
            "cu_seqlens_q": cu_q_cpu,
        }

    def customize(
        self,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        layout_query,
        layout_key,
        kwargs,
    ):
        pytest_golden = self.load_pytest_golden()
        quant_mode = int(kwargs.get("quant_mode") or 1)
        params = self.build_case_params(
            query,
            key,
            weights,
            query_dequant_scale,
            layout_query,
            layout_key,
            kwargs,
            include_weight_dtype=self.pytest_uses_weight_dtype(pytest_golden),
        )
        batch_module = self.load_batch_consistency()
        with batch_module.CaseRandomContext(kwargs):
            data = pytest_golden.generate_qliv2_test_data(params, generate_golden=False)
        batch_module.normalize_indexer_inputs(data, kwargs, "QLI_V2", quantized=True)
        for name, dst, src_name in (
            ("query", query, "query"),
            ("key", key, "key"),
            ("weights", weights, "weights"),
            ("query_dequant_scale", query_dequant_scale, "query_dequant_scale"),
            ("key_dequant_scale", key_dequant_scale, "key_dequant_scale"),
            ("cu_seqlens_q", cu_seqlens_q, "cu_seqlens_query"),
            ("cu_seqlens_k", cu_seqlens_k, "cu_seqlens_key"),
            ("seqused_q", seqused_q, "seqused_q"),
            ("seqused_k", seqused_k, "seqused_k"),
            ("cmp_residual_k", cmp_residual_k, "cmp_residual_k_for_npu"),
            ("block_table", block_table, "block_table"),
            ("output_idx_offset", output_idx_offset, "output_idx_offset"),
        ):
            src = data.get(src_name)
            if (
                quant_mode == QUANT_MODE_MXFP4
                and name in ("query", "key")
                and isinstance(dst, np.ndarray)
                and "float4" in str(dst.dtype)
            ):
                src = self.unpack_mxfp4(src, pytest_golden.FP4_E2M1_VALUES)
            packed_dtype = torch.float8_e4m3fn if quant_mode == QUANT_MODE_MXFP8 and name in ("query", "key") else None
            self.copy_tensor(dst, src, name, packed_dtype)
        return data


INPUT_ADAPTER = QuantLightningIndexerV2InputAdapter()


def rebuild_qli_v2_compare_data(compare_context):
    return INPUT_ADAPTER.rebuild_compare_data(compare_context)


def zero_metadata(metadata):
    if torch.is_tensor(metadata):
        metadata.zero_()
    else:
        metadata[...] = 0


def generate_qli_v2_inputs(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
    **kwargs,
):
    """Populate pytest-derived inputs; metadata is filled by npu_preprocess."""
    if metadata is None:
        raise ValueError("QLI_V2 direct API CSV must reserve the metadata tensor slot")
    params = dict(kwargs)
    params.update(
        {
            "sparse_count": topk,
            "quant_mode": quant_mode,
            "max_seqlen_q": max_seqlen_q,
            "layout_query": layout_q,
            "layout_key": layout_k,
            "sparse_mode": mask_mode,
            "cmp_ratio": cmp_ratio,
            "return_value": return_value,
        }
    )
    data = INPUT_ADAPTER.customize(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        layout_q,
        layout_k,
        params,
    )
    zero_metadata(metadata)
    case_data = INPUT_ADAPTER.load_golden_store().CASE_DATA
    testcase_name = params.get("testcase_name")
    case_data.put(testcase_name, data)
    INPUT_ADAPTER.load_metadata_protocol().save_metadata_inputs(
        "quant_lightning_indexer_v2", testcase_name, data.get("metadata_input")
    )
    return data


def generate_aclnn_qli_v2_inputs(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    block_table,
    output_idx_offset,
    metadata,
    topk,
    quant_mode,
    max_seqlen_q,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    return_value,
    sparse_indices_out,
    sparse_values_out,
    **kwargs,
):
    """Map the ACLNN C signature to the canonical pytest input adapter."""
    del sparse_indices_out, sparse_values_out
    return generate_qli_v2_inputs(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=max_seqlen_q,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        return_value=return_value,
        **kwargs,
    )
