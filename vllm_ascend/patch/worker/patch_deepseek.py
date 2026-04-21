#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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
"""
Patch for DeepSeek-R1-Distill models to fix invalid output issues.

The issue is that DeepSeek-R1-Distill models use LlamaTokenizerFast with
special tokens like <|begin▁of▁sentence|> and <|end▁of▁sentence|> that
may not be properly handled by the default tokenizer processing in vLLM.

This patch ensures proper handling of these special tokens.
"""

import functools
from typing import Any, Callable

from vllm.logger import logger

# DeepSeek-R1-Distill model prefixes
DEEPSEEK_R1_DISTILL_PREFIXES = [
    "DeepSeek-R1-Distill-Qwen",
    "DeepSeek-R1-Distill-Llama",
]


def is_deepseek_r1_distill(model_name: str) -> bool:
    """Check if the model is a DeepSeek-R1-Distill variant."""
    if not model_name:
        return False
    return any(prefix in model_name for prefix in DEEPSEEK_R1_DISTILL_PREFIXES)


def patch_tokenizer_get_vocab(tokenizer_get_vocab: Callable) -> Callable:
    """
    Patch tokenizer.get_vocab() to ensure special tokens are properly included.
    
    DeepSeek-R1-Distill models use special tokens that may not be correctly
    handled by the default tokenizer processing.
    """
    @functools.wraps(tokenizer_get_vocab)
    def wrapper(self, *args, **kwargs):
        vocab = tokenizer_get_vocab(self, *args, **kwargs)
        
        # Ensure special tokens are in vocab for DeepSeek-R1-Distill models
        special_tokens = [
            "<|begin▁of▁sentence|>",
            "<|end▁of▁sentence|>",
            "<|User|>",
            "<|Assistant|>",
        ]
        
        for token in special_tokens:
            if token not in vocab:
                # Try to get token ID using convert_tokens_to_ids
                try:
                    token_id = self.convert_tokens_to_ids(token)
                    if token_id is not None and token_id != self.unk_token_id:
                        vocab[token] = token_id
                except Exception:
                    pass
        
        return vocab
    
    return wrapper


def apply_deepseek_patches():
    """Apply patches for DeepSeek-R1-Distill models."""
    try:
        from transformers import PreTrainedTokenizerFast

        # Patch get_vocab to ensure special tokens are properly handled
        original_get_vocab = PreTrainedTokenizerFast.get_vocab
        PreTrainedTokenizerFast.get_vocab = patch_tokenizer_get_vocab(original_get_vocab)

        logger.info("Applied DeepSeek-R1-Distill tokenizer patches")
    except Exception as e:
        logger.warning(f"Failed to apply DeepSeek-R1-Distill patches: {e}")


# Apply patches when module is imported
apply_deepseek_patches()
