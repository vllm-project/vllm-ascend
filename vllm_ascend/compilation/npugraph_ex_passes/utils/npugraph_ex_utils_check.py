#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
#

from torch._inductor.pattern_matcher import Match
from vllm.logger import logger


def extra_stream_scope_check(match: Match) -> bool:
    """
    Checks if all nodes in the same stream.
    """
    non_default_streams = set()
    has_default = False

    for node in match.nodes:
        if node.op == "call_function":
            current_stream = node.meta.get("stream_label")
            if current_stream is None:
                has_default = True
            else:
                non_default_streams.add(current_stream)
                if len(non_default_streams) > 1:
                    logger.debug(
                        f"Cross-stream operation detected in pattern match for AddRMSNormQuant. "
                        f"Multiple streams found: {non_default_streams}. "
                        f"Fusion is not supported for cross-stream operations."
                    )
                    return False

    if has_default and len(non_default_streams) > 0:
        logger.debug(
            f"Cross-stream operation detected in pattern match for AddRMSNormQuant. "
            f"Multiple streams found: {non_default_streams}. "
            f"Fusion is not supported for cross-stream operations."
        )
        return False

    return True


_register_patterns = set()


def check_and_register_fusion_pass(pattern_class: type, **kwargs):
    global _register_patterns
    eps = kwargs.get("eps", 1e-6)
    pattern_key = str(pattern_class.__name__) + str(eps)
    if pattern_key in _register_patterns:
        return

    pattern = pattern_class(**kwargs)
    try:
        pattern.register()
        _register_patterns.add(pattern_key)
    except RuntimeError as e:
        if "Duplicate pattern" in str(e):
            logger.warning(f"Pattern {pattern_class.__name__} eps {eps} has been registered")
            _register_patterns.add(pattern_key)
        else:
            raise e
