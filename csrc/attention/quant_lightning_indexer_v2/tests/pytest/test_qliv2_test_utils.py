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

from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import pandas as pd
import pytest
from qliv2_parameter_normalization import normalize_qliv2_params
from qliv2_test_utils import (
    PARAM_NAMES,
    QliV2CaseSelector,
    QliV2ResultWriter,
    ensure_comparison_passed,
)


def test_case_selector_preserves_requested_order(tmp_path):
    for name in ("case_10.pt", "case_2.pt", "case_1.pt"):
        (tmp_path / name).touch()

    natural = QliV2CaseSelector.resolve(tmp_path)
    assert [Path(path).name for path in natural] == ["case_1.pt", "case_2.pt", "case_10.pt"]

    indexed = QliV2CaseSelector.resolve(tmp_path, case_indexes="3,1,2-2")
    assert [Path(path).name for path in indexed] == ["case_10.pt", "case_1.pt", "case_2.pt"]

    named = QliV2CaseSelector.resolve(tmp_path, case_names="case_2,case_1.pt")
    assert [Path(path).name for path in named] == ["case_2.pt", "case_1.pt"]


def test_case_selector_rejects_ambiguous_or_invalid_selection(tmp_path):
    (tmp_path / "case_1.pt").touch()
    with pytest.raises(ValueError, match="cannot be specified together"):
        QliV2CaseSelector.resolve(tmp_path, case_names="case_1", case_indexes="1")
    with pytest.raises(ValueError, match="out of range"):
        QliV2CaseSelector.resolve(tmp_path, case_indexes="2")


def test_result_writer_uses_readable_name_and_migrates_legacy_result(tmp_path):
    params = list(range(len(PARAM_NAMES)))
    params[PARAM_NAMES.index("qk_dtype")] = "INT8"
    params[PARAM_NAMES.index("layout_query")] = "BSND"
    params[PARAM_NAMES.index("layout_key")] = "PA_BBND"
    name = QliV2ResultWriter.case_name(params)
    assert name == QliV2ResultWriter.case_name(params)
    assert name == ("QLI_B0_S11_S22_N15_N26_D7_BSND_PA_BBND_INT8_QM20_SM24_CR30_K23_RV31")
    assert (
        QliV2ResultWriter.case_name(
            params,
            explicit_name="quant li/default:a5 v2",
        )
        == "quant_li_default_a5_v2"
    )

    row = QliV2ResultWriter.row(name, params, "Pass", 100.0)
    output = tmp_path / "result.xlsx"
    legacy = pd.DataFrame([{key: value for key, value in row.items() if key != "return_value"}])
    legacy.to_excel(output, index=False)

    QliV2ResultWriter.append(output, row)
    result = pd.read_excel(output)
    assert list(result.columns) == list(row.keys())
    assert len(result) == 2
    assert result.iloc[1]["return_value"] == params[PARAM_NAMES.index("return_value")]


def test_comparison_failure_raises_serializable_assertion():
    ensure_comparison_passed("case_pass", "Pass", 100.0)

    with pytest.raises(AssertionError, match="case_index_fail.*index result=Failed") as caught:
        ensure_comparison_passed("case_index_fail", "Failed", 97.5)
    restored = ForkingPickler.loads(ForkingPickler.dumps(caught.value))
    assert str(restored) == str(caught.value)

    with pytest.raises(AssertionError, match="case_value_fail.*value result=Failed"):
        ensure_comparison_passed("case_value_fail", "Pass", 100.0, "Failed", 90.0)


def test_normalize_legacy_params_adds_weight_dtype():
    params = list(range(32))
    params[10] = "INT8"
    params[11] = "FP16"

    normalized = normalize_qliv2_params(params)

    assert len(normalized) == 33
    assert normalized[:11] == tuple(params[:11])
    assert normalized[11] == params[11]
    assert normalized[12:] == tuple(params[11:])
