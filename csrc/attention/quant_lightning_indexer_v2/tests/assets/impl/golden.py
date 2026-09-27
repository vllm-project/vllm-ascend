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

"""CPU Golden adapter for QuantLightningIndexer V2 TTK cases."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

PYTEST_MODULE_NAME = "qli_v2_pytest_golden"
PYTEST_MODULE_FILE = "quant_lightning_indexer_v2_golden.py"


class CaseDataStore:
    """Share pytest data in-process and return compact metadata API inputs."""

    def __init__(self):
        self.case_data = {}
        self.active_testcase_name = None

    def clear(self):
        self.case_data.clear()
        self.active_testcase_name = None

    def put(self, testcase_name, data):
        if testcase_name is not None:
            self.case_data[str(testcase_name)] = data

    def get(self, testcase_name):
        if testcase_name is None:
            return None
        return self.case_data.get(str(testcase_name))

    def discard(self, data):
        for testcase_name, stored in tuple(self.case_data.items()):
            if stored is data:
                self.case_data.pop(testcase_name, None)


CASE_DATA = CaseDataStore()


def load_pytest_golden():
    """Load the pytest CPU reference only when the Golden stage needs it."""
    if PYTEST_MODULE_NAME in sys.modules:
        return sys.modules[PYTEST_MODULE_NAME]
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    path = pytest_dir / PYTEST_MODULE_FILE
    inserted = str(pytest_dir) not in sys.path
    if inserted:
        sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(PYTEST_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[PYTEST_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        sys.modules.pop(PYTEST_MODULE_NAME, None)
        raise RuntimeError(
            "Failed to load QuantLightningIndexer V2 pytest Golden module; "
            f"module={path.resolve()}; original error: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if inserted:
            sys.path.remove(str(pytest_dir))


def get_case_data(testcase_name):
    return CASE_DATA.get(testcase_name)


def materialize_golden(data):
    if data.get("cpu_result") is None:
        load_pytest_golden().generate_cpu_golden(data)
    return data


def activate_case_data(testcase_name):
    data = CASE_DATA.get(testcase_name)
    if data is None:
        raise RuntimeError("QuantLightningIndexer V2 Golden requires pytest data from the input stage")
    CASE_DATA.active_testcase_name = str(testcase_name)
    return materialize_golden(data)


def get_compare_data(testcase_name):
    if testcase_name is None:
        testcase_name = CASE_DATA.active_testcase_name
    if testcase_name is None:
        return None
    data = CASE_DATA.get(testcase_name)
    return None if data is None else materialize_golden(data)


def set_compare_data(testcase_name, data):
    name = str(testcase_name)
    CASE_DATA.active_testcase_name = name
    CASE_DATA.case_data[name] = data


def discard_compare_data(data):
    CASE_DATA.discard(data)
    CASE_DATA.active_testcase_name = None


def cpu_quant_lightning_indexer_v2(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    return_value=0,
    testcase_name=None,
    **kwargs,
):
    """Materialize Golden from the exact pytest data produced by input."""
    del query, key, weights, query_dequant_scale, key_dequant_scale
    del topk, quant_mode, kwargs
    data = activate_case_data(testcase_name)
    if int(return_value):
        sparse_value = data["cpu_topk_value"]
    else:
        sparse_value = torch.zeros(0, dtype=data["topk_value"].dtype)
    return data["cpu_result"], sparse_value


def cpu_aclnn_qli_v2(
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
    testcase_name=None,
    **kwargs,
):
    """Return the pytest Golden for the ACLNN C API parameter order."""
    del (
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        max_seqlen_q,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        sparse_indices_out,
    )
    sparse_indices, sparse_values = cpu_quant_lightning_indexer_v2(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        return_value=return_value,
        testcase_name=testcase_name,
        **kwargs,
    )
    if not int(return_value):
        if sparse_values_out is None:
            raise ValueError("ACLNN QLI_V2 requires the sparseValuesOut tensor slot")
        if torch.is_tensor(sparse_values_out):
            sparse_values = torch.zeros(tuple(sparse_values_out.shape), dtype=sparse_values_out.dtype)
        else:
            sparse_values = np.zeros_like(np.asarray(sparse_values_out))
    return sparse_indices, sparse_values
