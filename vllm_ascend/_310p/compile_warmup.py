#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from collections.abc import Iterable
from typing import Any

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import logger

from vllm_ascend.utils import check_gdn_layer


def mtp_full_decode_only_skips_piecewise_compile_warmup(vllm_config: VllmConfig) -> bool:
    """Whether 310P MTP FULL_DECODE_ONLY should skip piecewise compile dummy runs.

    ACL graph only captures uniform spec-decode batches. Piecewise warmup at
    compile_range.end (e.g. 2048 tokens) exercises ChunkedPrefill + GDN prefill
    paths that remain eager at runtime and lack 310P GDN metadata patches.
    """
    if vllm_config.model_config.enforce_eager:
        return False
    spec = vllm_config.speculative_config
    if spec is None or spec.method != "mtp":
        return False
    if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return False
    return check_gdn_layer(vllm_config)


def build_piecewise_compile_warmup_sizes(
    vllm_config: VllmConfig,
    warmup_sizes: list[int],
    cg_capture_sizes: list[int],
    compile_ranges: Iterable[Any],
) -> list[int]:
    """Build token sizes for piecewise compile warmup on 310P."""
    if vllm_config.model_config.enforce_eager:
        return warmup_sizes

    warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]
    if mtp_full_decode_only_skips_piecewise_compile_warmup(vllm_config):
        if warmup_sizes:
            logger.info_once(
                "310P MTP FULL_DECODE_ONLY with GDN: skipping piecewise compile warmup "
                "for token sizes %s. Prefill remains eager; ACL graph capture covers "
                "uniform spec-decode batches only.",
                warmup_sizes,
            )
        return []

    all_sizes = set(cg_capture_sizes)
    all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
    for compile_range in compile_ranges:
        if not any(x in compile_range for x in all_sizes):
            warmup_sizes.append(compile_range.end)
    return warmup_sizes
