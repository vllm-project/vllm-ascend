# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Optional


# Optional schema-v2 cost table applied after normal profiling. Unset keeps the
# measured table unchanged. This path is non-sensitive and read only at startup.
ENV_COST_TABLE_OVERRIDE = "VLLM_DCUT_COST_TABLE_OVERRIDE"


@dataclass
class VerifyAdaptiveConfig:
    """Config for the verifier adaptive step-length controller.

    ``query_len = 1 (anchor) + draft_len``.

    Load from a JSON path passed as ``--speculative-adaptive-verify-config``
    (or the ``speculative_adaptive_verify_config`` key in
    ``--speculative-config``), via :meth:`from_json` / :meth:`from_dict`.
    Unknown keys are silently ignored.  See ``verify_adaptive.md`` and
    ``verify_adaptive_config.example.json``.
    """

    # batch-size axis
    warmup_batch_sizes: list[int] = field(default_factory=list)
    """Explicit levels.  Empty → step-2 range from *min_warmup_batch_size*."""

    min_warmup_batch_size: int = 2
    """Start of the auto-generated batch-size range."""

    max_warmup_batch_size: Optional[int] = None
    """Cap for auto-generated levels.  None → engine max_num_seqs."""

    # query-length axis
    query_len_step_per_req: int = 2
    """Step between query-len levels.  Baseline 1 is always prepended."""

    max_query_len_per_req: Optional[int] = None
    """None → num_speculative_tokens + 1."""

    min_query_len_per_req: int = 2
    """Start of the step-range (must be ≥ 2)."""

    # measurement
    warmup_seq_lens: int = 4096
    """Simulated KV-context length (seq_lens) used during cost profiling.
    A larger value makes attention cost closer to real long-context inference.
    Defaults to 4096 for multimodal / long-context DFlash runs."""

    n_warmup_iters: int = 3
    n_measure_iters: int = 5

    cost_table_dump_path: Optional[str] = None
    """Optional JSON path for exporting the profiled verifier cost table.
    The environment variable ``VLLM_DCUT_COST_TABLE_OUT`` overrides this."""

    enabled: bool = True

    @classmethod
    def from_json(cls, path: str) -> "VerifyAdaptiveConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "VerifyAdaptiveConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self, num_speculative_tokens: int) -> None:
        eff_max_q = (
            self.max_query_len_per_req
            if self.max_query_len_per_req is not None
            else num_speculative_tokens + 1
        )
        if self.min_query_len_per_req < 2:
            raise ValueError(
                "min_query_len_per_req must be >= 2 "
                "(baseline query_len=1 is added automatically)."
            )
        if self.query_len_step_per_req < 1:
            raise ValueError("query_len_step_per_req must be >= 1.")
        if self.min_query_len_per_req > eff_max_q:
            raise ValueError(
                f"min_query_len_per_req ({self.min_query_len_per_req}) > "
                f"effective max_query_len_per_req ({eff_max_q})."
            )
        if self.warmup_seq_lens < 1:
            raise ValueError("warmup_seq_lens must be >= 1.")
        if self.n_warmup_iters < 0:
            raise ValueError("n_warmup_iters must be >= 0.")
        if self.n_measure_iters < 1:
            raise ValueError("n_measure_iters must be >= 1.")
        if self.warmup_batch_sizes and any(
            bs < 1 for bs in self.warmup_batch_sizes
        ):
            raise ValueError("All warmup_batch_sizes entries must be >= 1.")
        if self.min_warmup_batch_size < 1:
            raise ValueError("min_warmup_batch_size must be >= 1.")
        if self.max_warmup_batch_size is not None and self.max_warmup_batch_size < 1:
            raise ValueError("max_warmup_batch_size must be >= 1.")
