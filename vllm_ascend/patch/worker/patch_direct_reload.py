# SPDX-License-Identifier: Apache-2.0
"""Ascend runtime-format checkpoint views for the upstream direct-reload API.

Target: PR #64336/#65281's private checkpoint-view planner only.
Remove when upstream exposes a backend checkpoint-view provider hook.
The implementation and all storage policy remain in vllm-ascend.
"""

from vllm_ascend.model_loader.weight_reload import install_direct_reload_patch

install_direct_reload_patch()
