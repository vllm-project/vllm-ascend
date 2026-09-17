# SPDX-License-Identifier: Apache-2.0
"""Perf layout on the product branch.

**NPU live throughput (C1–C6, serve scripts, perf_lib)** lives on::

    feat/runtime-guard-analysis
    vllm_ascend/runtime_guard/test/perf/

**CPU microbench kept here** (needs product ``RuntimeConfig`` /
``RuntimeGuardProcessor``)::

    test_refresh_config_cost.py   # TEST_MATRIX A0 / A1

Do not re-add NPU harness scripts to this product tree.
"""
