# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import inspect

from vllm_ascend.worker.v2.input_batch import AscendInputBatch


def test_ascend_input_batch_fields_are_keyword_only() -> None:
    parameters = inspect.signature(AscendInputBatch).parameters

    assert parameters["seq_lens_np"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["attn_state"].kind is inspect.Parameter.KEYWORD_ONLY
