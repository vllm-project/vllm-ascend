# SPDX-License-Identifier: MIT
# Adapted from the DeepSeek V4.1 reference inference/engram.py.
from dataclasses import dataclass

import numpy as np
import torch
from sympy import isprime  # type: ignore[import-untyped]

_HISTORY_SLAB_MIN_TOKENS = 16
_PAGE_WRITE_NUMPY_MIN_TOKENS = 16


def engram_history_metadata(metadata):
    """Read full-request SWA pages, before attention's local CP slicing.

    V4.1 CP exposes local queries directly and keeps the replicated request
    in global_metadata. Engram runs before token slicing, so all TP ranks
    must use the full request lengths.
    """
    request_metadata = getattr(metadata, "global_metadata", None)
    if request_metadata is None:
        request_metadata = metadata
    boundaries = getattr(request_metadata, "query_start_loc_cpu", None)
    block_table = getattr(request_metadata, "block_table_cpu", None)
    if boundaries is None or block_table is None:
        raise ValueError("Engram requires query_start_loc_cpu and block_table_cpu in request metadata")
    if boundaries.device.type != "cpu" or block_table.device.type != "cpu":
        raise ValueError("Engram request metadata mirrors must reside on CPU")
    return boundaries.long(), block_table, request_metadata.storage_block_size


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
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike
    collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash
    the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters
    beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers  # type: ignore[import-untyped]

    # a private-use char, so a token that is exactly one space survives Strip() instead
    # of
    # collapsing to the empty string and merging with unrelated tokens
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

    # the raw Rust tokenizer, matching what training decodes with (no
    # clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw
            # form
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
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """Derive one multiplier per (layer, lookback) from a per-layer RNG.

    Kept odd, and bounded so that `token_id * multiplier` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
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


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position uses `max_ngram_size - 1` n-grams, each split over `n_heads`.
    Each (n-gram size, head) pair owns its own prime-sized bucket range in the
    layer's table; the primes are drawn in order and never reused, which keeps the
    ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes = []
        seen: set[int] = set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


class PagedNgramHistory:
    """Mirror token IDs in the scheduler's physical pages, including prefixes.

    A speculative suffix can be overwritten without rolling back a mutable
    per-request tail. Hashes read only positions at or before the current query.
    CPU residency supplies routing metadata without a device synchronization.
    """

    def __init__(self, config, tokenizer):
        layout = EngramLayout.from_args(config)
        if layout is None:
            raise ValueError("PagedNgramHistory requires at least one Engram layer")
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(f"Engram compressed vocabulary mismatch: {vocab_size}")
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = token_map[config.engram_pad_id]
        self.image_token_id = config.image_token_id
        self.image_pad_token_id = getattr(
            config,
            "image_pad_token_id",
            self.image_token_id + 1,
        )
        self.primes = torch.tensor(layout.primes)
        sizes = self.primes.flatten(1)
        self.offsets = sizes.cumsum(-1) - sizes
        self.multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        self.lookback = layout.max_ngram_size
        self.pages: dict[int, torch.Tensor] = {}

    def update(self, input_ids, positions, request_ids, block_table, block_size):
        """All arguments are CPU tensors; page numbers come from full SWA KV."""
        if input_ids.numel() == 0:
            # Idle DP and empty prefill still follow the collective contract,
            # but there is no page or hash state to update.
            columns = (self.lookback - 1) * self.primes.shape[-1]
            return (
                torch.empty((0, self.primes.shape[0], columns), dtype=torch.int64, device="cpu"),
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
                    token = self.pages[page][previous % block_size]
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
                # Reachable pages must exist, just as in the row path. Inactive
                # rows never read past an image or unwritten-token barrier.
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
