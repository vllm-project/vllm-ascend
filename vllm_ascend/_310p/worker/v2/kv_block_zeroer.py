# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Model Runner V2 adapter for the existing 310P KV block zeroer."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm_ascend._310p.kv_block_zeroer import AscendKVBlockZeroer310

if TYPE_CHECKING:
    from vllm_ascend.worker.utils import AttentionGroup


class AscendKVBlockZeroer310V2(AscendKVBlockZeroer310):
    """Normalize the V2 flat block-size layout without changing V1."""

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        v1_kernel_block_sizes = [[block_size] for block_size in kernel_block_sizes]
        super().init_meta(
            attn_groups_iter,
            v1_kernel_block_sizes,
            cache_dtype,
            runner_only_attn_layers,
            static_forward_context,
        )
