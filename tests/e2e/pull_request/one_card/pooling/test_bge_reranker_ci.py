# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.e2e.models.test_bge_reranker_correctness import BGE_CONFIG_PATH, run_bge_reranker_check


def test_bge_reranker_correctness(tmp_path):
    run_bge_reranker_check(BGE_CONFIG_PATH, tp_size=1, report_dir=tmp_path)
