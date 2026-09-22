# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side n-gram history and gating for the Ascend Engram port.

The bucket layout and the hashing helpers come from upstream, so a checkpoint
lands on the same rows here as it does on the accelerators upstream supports.
Only the token history differs: upstream keeps it on the device next to the SWA
cache, while the Ascend runner hands this module CPU page boundaries and the
physical page table, and the mirror stays on the host.
"""

import numpy as np
import torch
from vllm.models.deepseek_v41.common.engram import (  # type: ignore[import-not-found]
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)

_HISTORY_SLAB_MIN_TOKENS = 16
_PAGE_WRITE_NUMPY_MIN_TOKENS = 16


def engram_enabled(text_config) -> bool:
    """Whether the checkpoint declares Engram n-gram layers."""

    return bool(getattr(text_config, "engram_layer_ids", None))


def valid_engram_token_mask(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_pad_token_id: int,
) -> torch.Tensor:
    """Exclude the complete V4.1 image region from n-gram history."""
    return (input_ids != image_token_id) & (input_ids != image_pad_token_id)


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
        if compressed_vocab_size != layout.compressed_vocab_size:
            raise ValueError(
                f"Compressed vocab size mismatch: built {compressed_vocab_size} from the "
                f"tokenizer, config expects {layout.compressed_vocab_size}; every hash "
                "multiplier derives from it, so the engram tables would be silently rehashed."
            )
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
