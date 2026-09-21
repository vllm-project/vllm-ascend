# SPDX-License-Identifier: Apache-2.0
"""Reduced GLM checkpoint builder for numerical-precision and performance gates.

This package builds layer-reduced ("cropped") copies of real GLM checkpoints
for vllm-ascend testing. It keeps production dimensions, expert counts and
quantization metadata, streams safetensors shards without loading the full
checkpoint, and records an auditable manifest next to the output.

It is deliberately split into a pure transformation core (CPU, stdlib only,
unit-tested under ``tests/ut/tools/glm_reduced/``) and thin vLLM boundary
runners (``run_logits_dump.py`` / ``run_perf.py``) that require an NPU runtime.
"""

TOOL_NAME = "vllm-ascend-glm-reduced"
TOOL_VERSION = "1.0.0"
MANIFEST_SCHEMA = 1
MANIFEST_FILENAME = "reduction_manifest.json"
