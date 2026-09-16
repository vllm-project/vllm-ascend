# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""D448/rope0 accuracy against the original independent CPU attention golden."""

from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

import check_valid_param
import mixed_quant_sparse_flash_mla_golden as golden
from mixed_quant_sparse_flash_mla_paramset import TEST_PARAMS
from batch import mixed_quant_sparse_flash_mla_process as process
import result_compare_method
import utils


# Mode 2 uses PA_BBND only. Mode 1 also exercises BSND/TND KV addressing.
# CSA: B4/S1=2/S2=4096/K768/N128 exercises repeated tiles and split scheduling.
# SWA and ORI_SPARSE: S1=17/S2=257 include query and final KV block tails.
CASES = (
    ("q1_csa_tnd_pa", 1, "CSA", "TND", "PA_BBND", True),
    ("q1_csa_bsnd_bsnd", 1, "CSA", "BSND", "BSND", True),
    ("q2_csa_tnd_pa", 2, "CSA", "TND", "PA_BBND", True),
    ("q2_csa_bsnd_pa_no_lse", 2, "CSA", "BSND", "PA_BBND", False),
    ("q1_swa_tnd_tnd", 1, "SWA", "TND", "TND", True),
    ("q1_swa_bsnd_pa_no_lse", 1, "SWA", "BSND", "PA_BBND", False),
    ("q2_swa_tnd_pa", 2, "SWA", "TND", "PA_BBND", True),
    ("q2_swa_bsnd_pa_no_lse", 2, "SWA", "BSND", "PA_BBND", False),
    ("q1_ori_sparse_bsnd_pa", 1, "ORI_SPARSE", "BSND", "PA_BBND", True),
    ("q2_ori_sparse_tnd_pa_no_lse", 2, "ORI_SPARSE", "TND", "PA_BBND", False),
)


def make_params(case):
    name, quant_mode, template, layout_q, layout_kv, return_lse = case
    params = {key: deepcopy(values[0]) for key, values in TEST_PARAMS["decode_first"].items()}
    params.update(
        Testcase_Name=f"rope0_d448_{name}",
        D=448,
        rope_head_dim=0,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        template_run_mode=template,
        B=4 if template == "CSA" else 2,
        S1=2 if template == "CSA" else 17,
        S2=4096 if template == "CSA" else 257,
        N1=128 if template == "CSA" else 64,
        K=768 if template == "CSA" else 0,
        K1=61 if template == "ORI_SPARSE" else None,
        cmp_ratio=4 if template == "CSA" else 1,
        cmp_mask_mode=3 if template == "CSA" else 0,
        return_softmax_lse=return_lse,
        # Use the original generator's full valid top-k, including its mask semantics.
        ori_kv_topk_mode="fullK" if template == "ORI_SPARSE" else "no",
        ori_sparse_indices_mode="full",
        batch_consistency=False,
    )
    if name == "q1_swa_tnd_tnd":
        # qSNumInOneBlock=1: each of three Q iterations has G=5 rows, below NZ16.
        params.update(N1=5, S1=3)
    return utils.fill_none_params(params)


def assert_external_contract(data):
    params, inputs, metadata = data["params"], data["op_input"], data["metadata_input"]
    assert params["D"] == metadata["head_dim"] == 448
    assert params["rope_head_dim"] == metadata["rope_head_dim"] == inputs["rope_head_dim"] == 0
    q_shape = (
        (params["B"], params["S1"], params["N1"], 448)
        if params["layout_q"] == "BSND"
        else (params["T1"], params["N1"], 448)
    )
    assert tuple(inputs["q"].shape) == q_shape
    assert tuple(data["cpu_output"].shape) == q_shape
    assert data["golden_state"]["ori_k_bnsd"].shape[-1] == 448
    if params["Testcase_Name"] == "rope0_d448_q2_csa_bsnd_pa_no_lse":
        assert inputs["ori_topk_length"] is inputs["cmp_topk_length"] is None
        assert (inputs["ori_mask_mode"], inputs["cmp_mask_mode"]) == (4, 3)
        assert inputs["return_softmax_lse"] is False
    kv_bytes = 480 if params["quant_mode"] == 1 else 456
    for prefix in ("ori", "cmp"):
        kv = inputs[f"{prefix}_kv"]
        if kv is None:
            assert prefix == "cmp" and params["template_run_mode"] != "CSA"
            continue
        assert kv.element_size() == 1
        assert kv.shape[-1] == kv_bytes
        if params["layout_kv"] == "PA_BBND":
            suffix = "1" if prefix == "ori" else "2"
            assert tuple(kv.shape) == (
                params[f"block_num{suffix}"], params[f"block_size{suffix}"], params["N2"], kv_bytes
            )
        if params["quant_mode"] == 2:
            assert params["layout_kv"] == "PA_BBND"
            # Physical block: 448 feature bytes/token, then 7 scales + 1 pad/token.
            assert kv_bytes == 448 + (448 // params["tile_size"]) + 1
    if params["template_run_mode"] == "ORI_SPARSE":
        indices = inputs["ori_sparse_indices"]
        assert indices.shape[-1] == 61
        valid = indices[indices >= 0]
        assert valid.numel() > 0
        assert torch.unique(valid // params["block_size1"]).numel() > 1
    return q_shape


def observe_metadata(monkeypatch):
    # Observe the actual custom Metadata result without changing its arguments/output.
    original = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla_metadata
    captured = []

    def call(*args, **kwargs):
        metadata = original(*args, **kwargs)
        captured.append(metadata)
        return metadata

    monkeypatch.setattr(torch.ops._C_ascend, "npu_mixed_quant_sparse_flash_mla_metadata", call)
    return captured


def check_metadata_coverage(captured, params):
    assert len(captured) == 1
    metadata = captured[0].detach().cpu()
    assert metadata.dtype == torch.int32 and metadata.numel() == 1024
    # c6240b268 smla_metadata_common.h: AIC=36, AIV=72, FA stride=9, FD stride=8;
    # FD_CORE_ENABLE_INDEX=0. csa_kernel.h dispatches FD only when this is nonzero.
    fd = metadata.flatten()[36 * 9 : 36 * 9 + 72 * 8].reshape(72, 8)
    fd_enabled = int((fd[:, 0] != 0).sum())
    high_perf_expected = (
        params["ori_mask_mode"] == 4 and params["cmp_mask_mode"] == 3
        and not params["return_softmax_lse"]
    )
    print("ROPE0_METADATA_COVERAGE", json.dumps({
        "case": params["Testcase_Name"], "fd_enabled_cores": fd_enabled,
        "batch_consistency": params["batch_consistency"],
        "split_g_expected": params["N1"] // params["N2"] > 64,
        "high_perf_expected": high_perf_expected,
    }), flush=True)
    # Upstream Metadata enables FD only in batch-consistency (runtime level 3).
    # Ordinary execution must not be mistaken for coverage of that path.
    if params["batch_consistency"]:
        assert fd_enabled > 0, "A batch-consistency FD case must execute FD reduction"
    else:
        assert fd_enabled == 0, "Ordinary execution must not enable batch-consistency FD"


@pytest.mark.ci
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_mixed_quant_sparse_flash_mla_rope0(case, monkeypatch):
    seed = 95000 + [entry[0] for entry in CASES].index(case[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    params = make_params(case)
    # Invalid input and missing custom bindings must fail, never become a skip.
    check_valid_param.check_valid_param(params)
    data = golden.gen_data(params)
    expected_shape = assert_external_contract(data)
    captured = observe_metadata(monkeypatch)
    actual, expected, expected_lse, actual_lse = process.test_mqsmla_quant_process_ci(data)
    assert tuple(actual.shape) == expected_shape
    assert actual.dtype == torch.bfloat16
    result = result_compare_method.check_result(expected, actual)
    assert result[0] == "Pass", f"attention comparison failed: {result}"
    if params["return_softmax_lse"]:
        assert expected_lse is not None
        assert actual_lse.numel() == params["B"] * params["S1"] * params["N1"]
        result = result_compare_method.check_result(expected_lse, actual_lse)
        assert result[0] == "Pass", f"LSE comparison failed: {result}"
    else:
        assert expected_lse is None
        assert tuple(actual_lse.shape) == (0,)
        assert actual_lse.numel() == 0
    # Always complete the real output/LSE comparisons before coverage assertions.
    check_metadata_coverage(captured, params)
