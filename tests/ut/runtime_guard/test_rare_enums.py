#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CI gate: Python rare feature ids must match rare_feed_native.c."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_ascend.observability.runtime_guard.detector.rare import rare_enums, rare_native


def test_feat_input_loaded_from_c():
    feat, inp = rare_enums.build_feat_input()
    assert feat == rare_enums.FEAT
    assert inp == rare_enums.INPUT
    assert feat["lp1"] == 0
    assert feat["answer"] == 8
    assert inp["lp1"] == 0
    assert inp["bad_utf8"] == 14
    assert len(feat) == 9
    assert len(inp) == 15


def test_rare_native_tables_match_c():
    rare_enums.assert_enums_match_c(rare_native._FEAT, rare_native._INPUT)


def test_parse_rejects_missing_enumerator(tmp_path: Path):
    stub = tmp_path / "rare_feed_native.c"
    stub.write_text(
        "enum { FT_LP1 = 0, FT_LPC = 1 };\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="missing enumerators"):
        rare_enums.build_feat_input(c_path=stub)


def test_parse_rejects_unmapped_c_enumerator(tmp_path: Path):
    src = Path(rare_enums._C_PATH).read_text(encoding="utf-8")
    # Inject a new FT_* that the Python name map does not know.
    patched = src.replace("FT_ANSWER = 8", "FT_ANSWER = 8,\n    FT_EXTRA = 9")
    stub = tmp_path / "rare_feed_native.c"
    stub.write_text(patched, encoding="utf-8")
    with pytest.raises(KeyError, match="name map missing"):
        rare_enums.build_feat_input(c_path=stub)
