# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned layerwise cache planners, loaded only when pooling is enabled."""

from types import MappingProxyType
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import LayerwiseCachePlanner


CACHE_REUSE_PLANNERS = MappingProxyType(
    dict.fromkeys(
        ("glm5_next", "glm5_next_text"),
        "vllm_ascend.models.glm5next.cache_reuse.Glm5NextCachePlanner",
    )
)


def get_cache_reuse_planner(vllm_config: VllmConfig) -> type["LayerwiseCachePlanner"] | None:
    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    model_type = getattr(text_config, "model_type", None)
    path = CACHE_REUSE_PLANNERS.get(model_type)
    return resolve_obj_by_qualname(path) if path is not None else None
