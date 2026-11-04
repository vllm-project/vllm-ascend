# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from transformers import TokenizersBackend
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.hf import HfTokenizer, get_cached_tokenizer
from vllm.tokenizers.registry import TokenizerRegistry


class DeepSeekR1DistillLlamaTokenizer:
    """Preserve the checkpoint's ByteLevel tokenizer under Transformers 5.

    The checkpoint declares LlamaTokenizerFast, which Transformers 5 maps to
    a SentencePiece-style LlamaTokenizer. Loading the serialized backend
    directly preserves its pre-tokenizer, decoder, and special-token handling.
    This opt-in mode leaves the default tokenizer selection unchanged.
    """

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        download_dir: str | None = None,
        **kwargs,
    ) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(path_or_repo_id, *args, cache_dir=download_dir, **kwargs)
        return get_cached_tokenizer(tokenizer)


def register_tokenizers() -> None:
    mode = "deepseek_r1_distill_llama"
    if mode not in TokenizerRegistry.tokenizers:
        TokenizerRegistry.register(mode, "vllm_ascend.tokenizers", "DeepSeekR1DistillLlamaTokenizer")
    if mode not in RENDERER_REGISTRY.renderers:
        RENDERER_REGISTRY.register(mode, "vllm.renderers.hf", "HfRenderer")
