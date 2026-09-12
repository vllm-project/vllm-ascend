#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""Gemma4 MTP acceptance guard test.

Guards the per-position acceptance rate of Gemma4 MTP speculative decoding on
Ascend 950 (A5). Regression classes caught by this test include draft
attention metadata wiring (e.g. per-group block table assignment) and KV
sharing setup, which collapse pos1/pos2 acceptance while leaving the output
text correct.

Run `pytest tests/e2e/pull_request/one_card/spec_decode/test_gemma4_mtp.py`.
"""

import os

import pytest
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory
from vllm_ascend.device.device_config import is_950

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_HEADER = "Complete the following Python function. Only output the function body, no explanation.\n\n"
_TASKS = [
    (
        "def binary_search(arr, target):",
        '    """Return the index of target in sorted array arr, or -1 if not found."""',
    ),
    ("def merge_sort(arr):", '    """Sort the array arr in ascending order and return it."""'),
    (
        "def lru_cache_get(cache, key):",
        '    """Return value for key from an OrderedDict-backed LRU cache, or None. Mark as recently used on hit."""',
    ),
    (
        "def is_balanced(s):",
        '    """Return True if the string s has balanced parentheses/brackets/braces, else False."""',
    ),
    ("def flatten(nested):", '    """Yield elements from an arbitrarily nested list, depth-first."""'),
    ("def quick_sort(arr):", '    """Sort the array arr in ascending order using quicksort and return it."""'),
    (
        "def reverse_list(head):",
        '    """Reverse a singly linked list given its head node and return the new head."""',
    ),
    ("def fib(n):", '    """Return the n-th Fibonacci number iteratively (no recursion)."""'),
    (
        "def is_palindrome(s):",
        '    """Return True if s is a palindrome ignoring case and non-alphanumeric characters."""',
    ),
    ("def word_frequency(text):", '    """Return a dict mapping each lowercased word in text to its count."""'),
    ("def transpose(matrix):", '    """Return the transpose of a 2D matrix given as a list of lists."""'),
    ("def clamp(value, low, high):", '    """Clamp value into the inclusive range [low, high]."""'),
    (
        "def unique_preserve_order(items):",
        '    """Return items with duplicates removed, preserving first-occurrence order."""',
    ),
    (
        "def sum_of_digits(n):",
        '    """Return the sum of the decimal digits of the non-negative integer n."""',
    ),
    ("def capitalize_words(s):", '    """Capitalize the first letter of every word in s and return the result."""'),
]
# Fifteen distinct greedy trajectories keep the per-position draft sample
# large enough that a few borderline token flips cannot move the measured
# acceptance by more than a couple of points.
_EXAMPLE_PROMPTS = [_HEADER + f"{sig}\n{doc}" for sig, doc in _TASKS]

MODELS = ["google/gemma-4-31B-it"]
DRAFT_MODEL = "google/gemma-4-31B-it-assistant"

# Measured baseline is ~0.98/0.96/0.90 (multiple runs on 950DT); golden is
# set a few points below to absorb machine variance, and the assertion below
# tolerates another 0.06 shortfall per position.
GOLDEN_ACCEPTANCE = [0.94, 0.90, 0.84]


@pytest.mark.skipif(not is_950(), reason="Gemma4 MTP requires Ascend 950 (A5)")
@pytest.mark.parametrize("model_name", MODELS)
def test_gemma4_mtp_acceptance_tp1(model_name):
    with VllmRunner(
        model_name,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        disable_log_stats=False,
        speculative_config={
            "method": "mtp",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 3,
        },
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[20]),
    ) as spec_vllm_model:
        _ = spec_vllm_model.generate_greedy(_EXAMPLE_PROMPTS, 256)
        metrics = spec_vllm_model.model.get_metrics()

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * 3
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    assert num_drafts > 0, "no speculative drafts were executed"
    acceptance_per_pos = [accepted / num_drafts for accepted in num_accepted_tokens_per_pos]

    match = all((a >= b) or (b - a < 0.06) for a, b in zip(acceptance_per_pos, GOLDEN_ACCEPTANCE))
    assert match, (
        f"acceptance_per_pos {acceptance_per_pos} does not match golden {GOLDEN_ACCEPTANCE} (num_drafts={num_drafts})"
    )
    cleanup_dist_env_and_memory()
