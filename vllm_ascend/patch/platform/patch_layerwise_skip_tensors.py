# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exclude Ascend runtime buffers/shared parameters from vLLM reload accounting.

Mutate the existing sets: upstream layerwise.py imports them by reference.
Exclusion preserves storage, not values across sleep. Frozen parameters must use
ModelConfig.sleep_preserve_parameter_names; Worker handles their CPU backup.
Remove these exclusions when upstream supports model/shared-tensor accounting.
"""

from vllm.model_executor.model_loader.reload import meta

_RUNTIME_TENSORS = {
    "expert_ids_per_ep_rank",
    "log2phy",
    "moe_load",
    "load_counter",
    "tid2eid",
    "bias_vl",
    "attn_sink",
    "cos_sin_cache",
    "cos_cached",
    "sin_cached",
    "full_rope_cos",
    "full_rope_sin",
    "image_start",
    "image_end",
    "image_newline",
    "image_pad",
    "_sentinel_table",
    "_sleep_norm_weight",
}
meta.SKIP_LOAD_TENSORS.update(_RUNTIME_TENSORS)
meta.SKIP_TENSORS.update(_RUNTIME_TENSORS)
