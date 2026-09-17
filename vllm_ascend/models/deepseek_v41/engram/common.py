# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""N-gram hashing and gating, independent of where the tables are stored.

The hash helpers mirror ``vllm.models.deepseek_v41.common.engram`` so the same
checkpoint hashes to the same rows on either accelerator.
"""

import numpy as np
import torch

_HISTORY_SLAB_MIN_TOKENS = 16
_PAGE_WRITE_NUMPY_MIN_TOKENS = 16


def engram_enabled(text_config) -> bool:
    """Whether the checkpoint declares Engram n-gram layers."""

    return bool(getattr(text_config, "engram_layer_ids", None))


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


def valid_engram_token_mask(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_pad_token_id: int,
) -> torch.Tensor:
    """Exclude the complete V4.1 image region from n-gram history."""
    return (input_ids != image_token_id) & (input_ids != image_pad_token_id)


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize
    alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE"
    all hash the same way. The compressed size matters beyond bounds checking:
    every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers  # type: ignore[import-untyped]

    # A private-use char, so a token that is exactly one space survives
    # Strip() instead of collapsing to the empty string and merging with
    # unrelated tokens.
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

    # The raw Rust tokenizer, matching what training decodes with
    # (no clean_up_tokenization_spaces).
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it
            # by its raw form.
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
    """One multiplier per (layer, lookback), from a per-layer RNG so layers
    hash differently. Kept odd and bounded so `token_id * multiplier` cannot
    overflow int64.
    """
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
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max), each
    split over `n_heads` heads. Every (n-gram size, head) pair owns its own
    prime-sized bucket range in the layer's table; the primes are drawn in
    order and never reused, which keeps the ranges disjoint.

    The compressed vocab size is rebuilt from the tokenizer rather than read
    from the checkpoint, which is the only source this port has; every hash
    multiplier derives from it.
    """

    def __init__(self, config) -> None:
        self.layer_ids: tuple[int, ...] = tuple(config.engram_layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(config.engram_num_embeddings)
        self.max_ngram_size: int = config.engram_max_ngram_size
        self.n_heads: int = config.engram_n_heads
        self.head_dim: int = config.engram_head_dim
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
        self.offsets = torch.tensor(np.array(offsets))  # [n_layers, n_hash_cols]

    @classmethod
    def from_config(cls, config) -> "EngramLayout | None":
        if not engram_enabled(config):
            return None
        return cls(config)


class PagedNgramHistory:
    """Mirror token IDs in the scheduler's physical pages, including prefixes.

    A speculative suffix can be overwritten without rolling back a mutable
    per-request tail. Hashes read only positions at or before the current query.
    CPU residency supplies routing metadata without a device synchronization.
    """

    def __init__(self, config, tokenizer):
        layout = EngramLayout.from_config(config)
        assert layout is not None, "Paged history requires at least one Engram layer"
        token_map, compressed_vocab_size = build_compressed_token_map(tokenizer)
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = token_map[layout.pad_token_id]
        self.image_token_id = config.image_token_id
        self.image_pad_token_id = getattr(
            config,
            "image_pad_token_id",
            self.image_token_id + 1,
        )
        self.primes = torch.tensor(layout.primes)
        self.offsets = layout.offsets
        self.multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, compressed_vocab_size)
        self.lookback = layout.max_ngram_size
        self.n_hash_cols = layout.n_hash_cols
        self.pages: dict[int, torch.Tensor] = {}

    def update(self, input_ids, positions, request_ids, block_table, block_size):
        """All arguments are CPU tensors; page numbers come from full SWA KV."""
        if input_ids.numel() == 0:
            # Idle DP and empty prefill still follow the collective contract,
            # but there is no page or hash state to update.
            return (
                torch.empty((0, self.primes.shape[0], self.n_hash_cols), dtype=torch.int64, device="cpu"),
                torch.empty(0, dtype=torch.bool, device="cpu"),
            )
        compressed = self.token_map[input_ids]
        mask = valid_engram_token_mask(
            input_ids,
            self.image_token_id,
            self.image_pad_token_id,
        )
        compressed = compressed.masked_fill(~mask, -1)
        # Materialize CPU lists once for sequential page writes and small-batch
        # history reads.
        compressed_list = compressed.tolist()
        position_list = positions.tolist()
        page_indices = block_table[request_ids, positions // block_size].tolist()
        if len(input_ids) < _PAGE_WRITE_NUMPY_MIN_TOKENS:
            for token, position, page in zip(compressed_list, position_list, page_indices):
                if page not in self.pages:
                    self.pages[page] = torch.full((block_size,), -1, dtype=torch.int64, device="cpu")
                self.pages[page][position % block_size] = token
        else:
            page_views: dict[int, np.ndarray] = {}
            for token, position, page in zip(compressed_list, position_list, page_indices):
                view = page_views.get(page)
                if view is None:
                    if page not in self.pages:
                        self.pages[page] = torch.full((block_size,), -1, dtype=torch.int64, device="cpu")
                    view = self.pages[page].numpy()
                    page_views[page] = view
                # Zero-copy CPU view avoids Torch dispatch per scalar write. Keep
                # input order so repeated physical slots retain last-write wins.
                view[position % block_size] = token
        history = torch.full((len(input_ids), self.lookback), self.pad_id, dtype=torch.int64, device="cpu")
        if len(input_ids) < _HISTORY_SLAB_MIN_TOKENS:
            # Slab construction dominates decode and small batches; retain the
            # page-row loop for this latency-sensitive path.
            for row, (position, request) in enumerate(zip(position_list, request_ids.tolist())):
                for shift in range(self.lookback):
                    previous = position - shift
                    if previous < 0:
                        break
                    page = block_table[request, previous // block_size].item()
                    page_tokens = self.pages.get(page)
                    if page_tokens is None:
                        # This replica never wrote that page: a prefix that was
                        # transferred from another instance (P/D split) or a
                        # recompute that has not reached it yet. There is no
                        # history to read, exactly like an unwritten slot.
                        break
                    token = page_tokens[previous % block_size]
                    if token < 0:
                        break
                    history[row, shift] = token
        else:
            active = torch.ones(len(input_ids), dtype=torch.bool, device="cpu")
            for shift in range(self.lookback):
                previous = positions - shift
                valid = active & (previous >= 0)
                if not bool(valid.any()):
                    break
                with torch.device("cpu"):
                    rows = torch.nonzero(valid, as_tuple=False).flatten()
                page_ids = block_table[request_ids[rows], previous[rows] // block_size]
                offsets = previous[rows] % block_size
                with torch.device("cpu"):
                    unique_pages, slab_indices = torch.unique(page_ids, return_inverse=True)
                # Inactive rows never read past an image or unwritten-token
                # barrier. Pages this replica never wrote (transferred prefix,
                # in-flight recompute) carry no token, so materialize them as
                # unwritten instead of failing the lookup.
                for page in unique_pages.tolist():
                    if page not in self.pages:
                        self.pages[page] = torch.full((block_size,), -1, dtype=torch.int64, device="cpu")
                slab = torch.stack([self.pages[page] for page in unique_pages.tolist()])
                values = slab[slab_indices, offsets]
                present = values >= 0
                history[rows[present], shift] = values[present]
                active[rows] = present
        products = history[:, None] * self.multipliers
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.lookback):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling[..., None] % self.primes[:, shift - 1])
        return torch.cat(hashes, -1) + self.offsets, mask


def engram_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply original-basis gating to a rotated residual and rotated value.

    ``hidden`` and ``key`` have shape [tokens, hc_mult, hidden_size].
    The saved rotation consists of identical diagonal blocks. Restore hidden
    in FP32; the value projection already includes the forward rotation.
    """
    dim = hidden.shape[-1]
    original = (hidden.float().unflatten(-1, (-1, rotation_block.shape[0])) @ rotation_block.float().T).flatten(-2)
    key = key.float()
    rstd = torch.rsqrt(original.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key.square().mean(-1) + eps)
    dot = (original * channel_weight.float() * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hidden.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
