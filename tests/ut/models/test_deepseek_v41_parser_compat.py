# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import json

from vllm_ascend.compat.deepseek_v41.parser import _dsml_arg_converter_v41


def test_dsml_argument_converter_preserves_strings_and_json_values():
    raw = (
        '<｜DSML｜ parameter name="query" string="true">hello world</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="limit" string="false">3</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="filters" string="false">["docs", "code"]</｜DSML｜ parameter>'
    )

    assert json.loads(_dsml_arg_converter_v41(raw, partial=False)) == {
        "query": "hello world",
        "limit": 3,
        "filters": ["docs", "code"],
    }


def test_partial_dsml_omits_incomplete_non_string_json():
    raw = (
        '<｜DSML｜ parameter name="query" string="true">hello</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="filters" string="false">["docs"'
    )

    assert json.loads(_dsml_arg_converter_v41(raw, partial=True)) == {"query": "hello"}
