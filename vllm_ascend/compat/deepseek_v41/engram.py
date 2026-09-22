# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared DeepSeek-V4.1 Engram hashing and runtime compatibility helpers.

This v0.29 backport contains the platform-independent pieces consumed by
vLLM Ascend plus the small base-class surface that the Ascend implementation
subclasses. Device-specific hashing and embedding implementations remain in
their owning platform integration.
"""

import weakref

import numpy as np
import torch
from torch import nn
from vllm.triton_utils import tl, triton

# Cache value for tokens that take no part in an n-gram (image spans).
DEAD_ID = -1


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (avoids a sympy import)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Return the smallest unused prime above ``start``."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map token IDs to normalized IDs used by the Engram n-gram hashes."""
    from tokenizers import Regex, normalizers  # type: ignore[import-untyped]

    # Preserve a token that is exactly one space through Strip().
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # Preserve partial UTF-8 byte tokens by their raw representation.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """Build deterministic, odd int64-safe hash multipliers per layer."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout:
    """Bucket layout of the n-gram hash tables."""

    def __init__(self, config) -> None:
        self.layer_ids: tuple[int, ...] = tuple(config.engram_layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(config.engram_num_embeddings)
        self.max_ngram_size: int = config.engram_max_ngram_size
        self.n_heads: int = config.engram_n_heads
        self.head_dim: int = config.engram_head_dim
        self.compressed_vocab_size: int = config.engram_compressed_vocab_size
        self.pad_token_id: int = config.engram_pad_token_id
        assert len(self.layer_ids) == len(self.num_embeddings)

        primes = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        self.primes: tuple[tuple[tuple[int, ...], ...], ...] = tuple(primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.offsets = torch.tensor(np.array(offsets))

    @classmethod
    def from_config(cls, config) -> "EngramLayout | None":
        if not getattr(config, "engram_layer_ids", None):
            return None
        return cls(config)


@triton.jit(do_not_specialize=["num_tokens"])
def _write_hash_cache_kernel(
    input_ids,
    token_map,
    dead_mask,
    slot_mapping,
    cache,
    num_tokens,
    input_stride,
    mask_stride,
    slot_stride,
    BLOCK_SIZE: tl.constexpr,
    dead_id,
):
    """Write compressed token ids to the slot-keyed history cache."""
    token_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    slot = tl.load(
        slot_mapping + token_idx * slot_stride,
        token_idx < num_tokens,
        other=-1,
    ).to(tl.int64)
    valid = (token_idx < num_tokens) & (slot >= 0)
    token = tl.load(input_ids + token_idx * input_stride, valid, other=0)
    value = tl.load(token_map + token, valid, other=0)
    dead = tl.load(dead_mask + token_idx * mask_stride, valid, other=False)
    value = tl.where(dead, dead_id, value)
    tl.store(cache + slot, value, valid)


class NgramHashState(nn.Module):
    """v0.29 base state used by the Ascend hash implementation.

    The NPU subclass owns the hash launch. This class retains the tokenizer
    setup and slot-cache lifecycle from the newer vLLM implementation.
    """

    def __init__(self, vllm_config, layout: EngramLayout, swa_cache_module: nn.Module) -> None:
        super().__init__()
        self.layout = layout
        self.swa_cache_module = swa_cache_module
        self.block_size: int = swa_cache_module.block_size
        self.lookback_depth: int = layout.max_ngram_size - 1
        self.use_slot_cache: bool = not vllm_config.use_v2_model_runner
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

        from transformers import AutoTokenizer

        model_config = vllm_config.model_config
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != layout.compressed_vocab_size:
            raise ValueError(
                f"Compressed vocab size mismatch: built {vocab_size} from the "
                f"tokenizer, config expects {layout.compressed_vocab_size}; "
                "every hash multiplier derives from it, so the engram tables "
                "would be silently rehashed."
            )
        self.pad_id = token_map[layout.pad_token_id]
        multipliers = compute_hash_multipliers(
            layout.layer_ids,
            layout.max_ngram_size,
            vocab_size,
        )
        self.register_buffer(
            "token_map",
            torch.tensor(token_map, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "primes",
            torch.tensor(layout.primes),
            persistent=False,
        )
        self.register_buffer("offsets", layout.offsets, persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)

    def ensure_cache(self) -> bool:
        """Size the optional slot-keyed cache after the SWA cache is bound."""
        kv_cache = self.swa_cache_module.kv_cache
        if kv_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if not self.use_slot_cache:
            return True
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv_cache:
            return True
        self._cache = torch.zeros(
            kv_cache.shape[0] * self.block_size,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._kv_cache_ref = weakref.ref(kv_cache)
        return True


class ParallelEngramEmbedding(nn.Module):
    """Compatibility base for the fully overridden Ascend embedding table."""

    pass
