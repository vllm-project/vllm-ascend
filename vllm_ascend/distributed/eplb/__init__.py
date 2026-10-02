# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend integration for the vLLM distributed EPLB runtime."""

# Set on an EPLBConfig instance when Ascend auto-selected the torch_gloo
# communicator because no HIXL binding was available. Unlike an explicit
# torch_gloo choice, the auto fallback also clamps STAIR migration limits.
AUTO_GLOO_FALLBACK_ATTRIBUTE = "_vllm_ascend_eplb_auto_gloo_fallback"
