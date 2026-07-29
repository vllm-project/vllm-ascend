"""Patch entrypoints for DSA sparse-cache offload.

DSA sparse offload is an Ascend-specific feature.  The long-term direction is
to keep vLLM source close to upstream and apply DSA integration from
vllm-ascend, following the existing platform/worker patch lifecycle.
"""
