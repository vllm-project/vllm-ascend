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

"""TTK result adapter for the QuantLightningIndexer V2 pytest TopK comparison."""

import importlib.util
import logging
import sys
import threading
from pathlib import Path

import numpy as np
import torch


class PytestV2TopKComparator:
    """Run the pytest V2 TopK compare with replay-safe data from the TestSpec."""

    def __init__(self):
        self.module = None
        self.lock = threading.Lock()

    def load_module(self):
        if self.module is not None:
            return self.module
        with self.lock:
            if self.module is not None:
                return self.module
            pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
            module_path = pytest_dir / "result_compare_method.py"
            module_name = "qli_v2_ttk_pytest_compare"
            inserted = str(pytest_dir) not in sys.path
            original_basic_config = logging.basicConfig
            if inserted:
                sys.path.insert(0, str(pytest_dir))
            try:
                logging.basicConfig = lambda *args, **kwargs: None
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self.module = module
            except Exception as exc:
                sys.modules.pop(module_name, None)
                raise RuntimeError(
                    "Failed to load QuantLightningIndexerV2 pytest compare; "
                    f"module={module_path.resolve()}; "
                    f"original error: {type(exc).__name__}: {exc}"
                ) from exc
            finally:
                logging.basicConfig = original_basic_config
                if inserted and str(pytest_dir) in sys.path:
                    sys.path.remove(str(pytest_dir))
        return self.module

    @staticmethod
    def to_torch(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        array = np.array(value, copy=True, order="C")
        dtype_name = str(array.dtype)
        custom_dtypes = {
            "bfloat16": (np.uint16, torch.bfloat16),
            "float8_e4m3fn": (np.uint8, torch.float8_e4m3fn),
            "float8_e5m2": (np.uint8, torch.float8_e5m2),
        }
        if dtype_name in custom_dtypes:
            storage_dtype, torch_dtype = custom_dtypes[dtype_name]
            storage = np.ascontiguousarray(array).view(storage_dtype)
            return torch.from_numpy(storage).view(torch_dtype).reshape(array.shape)
        return torch.from_numpy(array)

    @staticmethod
    def result_dict(result, stage):
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise ValueError(f"pytest {stage} returned invalid result: {result!r}")
        status, precision = result[:2]
        passed = str(status).strip().lower() == "pass"
        return {
            "pass": passed,
            "precision": float(precision),
            "error_info": None if passed else (f"pytest QuantLightningIndexerV2 {stage} returned {status!r}"),
        }

    def compare(self, *outputs, compare_data=None):
        if compare_data is None:
            raise ValueError("QuantLightningIndexerV2 pytest compare data is unavailable")
        if len(outputs) < 2 or len(outputs) % 2 != 0:
            return {
                "pass": False,
                "precision": "invalid",
                "error_info": "compare expects NPU outputs followed by golden outputs",
            }
        params = compare_data.get("params")
        topk_value = compare_data.get("topk_value")
        if params is None or topk_value is None:
            raise ValueError("QuantLightningIndexerV2 pytest compare data lacks params or topk_value")
        half = len(outputs) // 2
        npu_outputs = outputs[:half]
        golden_outputs = outputs[half:]
        if tuple(getattr(npu_outputs[0], "shape", ())) != tuple(getattr(golden_outputs[0], "shape", ())):
            return {
                "pass": False,
                "precision": "shape_mismatch",
                "error_info": (
                    "index output shape mismatch: "
                    f"npu={getattr(npu_outputs[0], 'shape', None)}, "
                    f"golden={getattr(golden_outputs[0], 'shape', None)}"
                ),
            }
        return_value = bool(params[-2])
        if return_value and half < 2:
            return {
                "pass": False,
                "precision": "missing_output",
                "error_info": "return_value is enabled but the NPU sparse-value output is missing",
            }
        npu_values = npu_outputs[1] if half > 1 else torch.empty(0)
        golden_values = golden_outputs[1] if half > 1 else torch.empty(0)
        if return_value and tuple(getattr(npu_values, "shape", ())) != tuple(getattr(golden_values, "shape", ())):
            return {
                "pass": False,
                "precision": "shape_mismatch",
                "error_info": (
                    "sparse-value output shape mismatch: "
                    f"npu={getattr(npu_values, 'shape', None)}, "
                    f"golden={getattr(golden_values, 'shape', None)}"
                ),
            }
        npu_indices = self.to_torch(npu_outputs[0])
        npu_values = self.to_torch(npu_values)
        golden_values = self.to_torch(golden_values)
        if return_value:
            npu_values, sort_order = npu_values.sort(dim=-1, descending=True)
            npu_indices = torch.gather(npu_indices, dim=-1, index=sort_order)
        golden_indices = self.to_torch(golden_outputs[0])
        topk_value = self.to_torch(topk_value)
        output_idx_offset = self.to_torch(compare_data.get("output_idx_offset"))
        golden_values_for_index = golden_values.detach().cpu().float().numpy()
        npu_values_for_index = npu_values.detach().cpu().float().numpy()
        module = self.load_module()
        index_result = module.check_result(
            golden_indices,
            npu_indices,
            topk_value,
            output_idx_offset,
            params,
            golden_values_for_index,
            npu_values_for_index,
        )
        results = [self.result_dict(index_result, "index compare")]
        if return_value:
            value_result = module.check_result_return_value(
                golden_values,
                npu_values,
                params,
                golden_indices,
                npu_indices,
                topk_value,
                output_idx_offset,
            )
            results.append(self.result_dict(value_result, "sparse-value compare"))
        return results


def load_batch_comparator():
    name = "qli_v2_ttk_batch_consistency"
    path = Path(__file__).with_name("batch_consistency.py")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COMPARATOR = PytestV2TopKComparator()


def compare(
    *outputs,
    compare_data=None,
    batch_consistency_id=None,
    batch_axis=None,
    batch_slice_info=None,
    batch_seed=None,
    compare_context=None,
):
    """Compare V2 TopK outputs with the canonical pytest policy."""
    results = COMPARATOR.compare(*outputs, compare_data=compare_data)
    if not isinstance(results, list) or not all(result["pass"] for result in results):
        return results
    batch_result = (
        load_batch_comparator()
        .IndexerBatchOutputComparator("QLI_V2")
        .compare(
            outputs[0],
            batch_consistency_id,
            batch_axis,
            batch_slice_info,
            batch_seed,
            compare_context,
        )
    )
    if batch_result is not None:
        results.append(batch_result)
    return results
