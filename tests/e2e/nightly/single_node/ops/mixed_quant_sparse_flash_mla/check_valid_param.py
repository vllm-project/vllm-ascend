#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import math
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def check_valid_param(params):
    if params.get("q_type") not in [torch.bfloat16]:
        raise ValueError("q_type should be: bfloat16")

    if params.get("layout_q") not in ["BSND", "TND"]:
        raise ValueError(f"不支持的Q shape: {params.get('layout_q')}")

    if params.get("layout_kv") not in ["BSND", "PA_BBND", "TND"]:
        raise ValueError(f"不支持的KV shape: {params.get('layout_kv')}")

    if params.get("template_run_mode") not in [
        "CSA",
        "HCA",
        "SWA",
        "ORI_SPARSE",
        "ORI_CMP_SPARSE",
    ]:
        raise ValueError(
            f"不支持的template_run_mode: {params.get('template_run_mode')}"
        )

    template_run_mode = params.get("template_run_mode")
    ori_mask_mode = params.get("ori_mask_mode")
    cmp_mask_mode = params.get("cmp_mask_mode")

    if template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE"):
        if ori_mask_mode != 0 and ori_mask_mode != 3 and ori_mask_mode != 4:
            raise ValueError(
                f"ORI_SPARSE/ORI_CMP_SPARSE模式下ori_mask_mode仅支持0/3/4, 当前为{ori_mask_mode}"
            )
        if cmp_mask_mode != 0 and cmp_mask_mode != 3:
            raise ValueError(
                f"ORI_SPARSE/ORI_CMP_SPARSE模式下cmp_mask_mode仅支持0/3, 当前为{cmp_mask_mode}"
            )

    B = params.get("B")
    cmp_ratio = params.get("cmp_ratio")

    if params.get("layout_q") == "TND":
        cu_seqlens_q = params.get("cu_seqlens_q")
        T1 = params.get("T1")
        if cu_seqlens_q is None or T1 is None:
            raise ValueError("layout_q为TND时，cu_seqlens_q和T1不能为None")
        if cu_seqlens_q[-1] != T1:
            raise ValueError(
                f"cu_seqlens_q最后一维({cu_seqlens_q[-1]})与T1({T1})不一致"
            )
        if len(cu_seqlens_q) != B + 1:
            raise ValueError(f"cu_seqlens_q长度({len(cu_seqlens_q)})应等于B+1({B + 1})")

    if params.get("layout_kv") == "TND":
        cu_seqlens_ori_kv = params.get("cu_seqlens_ori_kv")
        if cu_seqlens_ori_kv is None:
            raise ValueError("layout_kv为TND时，cu_seqlens_ori_kv不能为None")
        T2 = cu_seqlens_ori_kv[-1]
        if len(cu_seqlens_ori_kv) != B + 1:
            raise ValueError(
                f"cu_seqlens_ori_kv长度({len(cu_seqlens_ori_kv)})应等于B+1({B + 1})"
            )
        if not all(
            cu_seqlens_ori_kv[i] <= cu_seqlens_ori_kv[i + 1]
            for i in range(len(cu_seqlens_ori_kv) - 1)
        ):
            raise ValueError(f"cu_seqlens_ori_kv必须单调递增: {cu_seqlens_ori_kv}")

        cu_seqlens_cmp_kv = params.get("cu_seqlens_cmp_kv")
        if cu_seqlens_cmp_kv is not None:
            T3 = cu_seqlens_cmp_kv[-1]
            if len(cu_seqlens_cmp_kv) != B + 1:
                raise ValueError(
                    f"cu_seqlens_cmp_kv长度({len(cu_seqlens_cmp_kv)})应等于B+1({B + 1})"
                )
            if not all(
                cu_seqlens_cmp_kv[i] <= cu_seqlens_cmp_kv[i + 1]
                for i in range(len(cu_seqlens_cmp_kv) - 1)
            ):
                raise ValueError(f"cu_seqlens_cmp_kv必须单调递增: {cu_seqlens_cmp_kv}")
            if cmp_ratio is not None and cmp_ratio != 1:
                expected_T3 = math.floor(T2 / cmp_ratio)
                if T3 != expected_T3:
                    raise ValueError(
                        f"cu_seqlens_cmp_kv最后一维({T3})与预期T3({expected_T3})不一致, T2={T2}, cmp_ratio={cmp_ratio}"
                    )
