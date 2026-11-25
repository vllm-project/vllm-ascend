# SPDX-License-Identifier: Apache-2.0
"""Collect BF16 KV samples and query statistics for NOVA-KV calibration."""

from pathlib import Path

import regex as re
import torch
from vllm.distributed import get_tensor_model_parallel_rank


class NovaCapture:
    """Keep a bounded reservoir per layer and TP rank on the host."""

    def __init__(self, layer_name, options):
        self.layer_id = int(re.search(r"layers\.(\d+)", layer_name).group(1))
        self.root = Path(options.capture_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank = get_tensor_model_parallel_rank()
        self.limit = options.sample_tokens
        self.seen = 0
        self.generator = torch.Generator().manual_seed(1024 + self.layer_id)
        self.key = self.value = self.query_moment = None

    def add(self, query, key, value):
        q, k, v = [x.detach().float().cpu() for x in (query, key, value)]
        heads, dim = k.shape[1:]
        q = q.reshape(q.shape[0], heads, -1, dim).double()
        moment = torch.einsum("thgd,thgf->hdf", q, q)
        self.query_moment = moment if self.query_moment is None else self.query_moment + moment
        if self.key is None:
            self.key = torch.empty((self.limit, heads, dim), dtype=torch.bfloat16)
            self.value = torch.empty_like(self.key)
        for row in range(k.shape[0]):
            position = self.seen
            if position >= self.limit:
                position = int(torch.randint(self.seen + 1, (1,), generator=self.generator))
            if position < self.limit:
                self.key[position], self.value[position] = k[row], v[row]
            self.seen += 1
        count = min(self.limit, self.seen)
        path = self.root / f"layer_{self.layer_id:02d}_rank_{self.rank}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "key": self.key[:count],
                "value": self.value[:count],
                "query_moment": self.query_moment,
                "seen": self.seen,
            },
            temporary,
        )
        temporary.replace(path)
