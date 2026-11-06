# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-mode contract shared by the context-parallel attention backends.

The CP backends used to infer graph semantics from two booleans that cannot
distinguish piecewise from full graphs: ``cudagraph_mode.has_full_cudagraphs()``
and ``_EXTRA_CTX.capturing``. ``CPGraphSpec`` makes the mode that will replay a
batch explicit, together with the padded extents that belong to it.

See ``docs/source/user_guide/feature_guide/context_parallel.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

DSA_FAMILY = "dsa"


class CPGraphMode(str, Enum):
    """Graph mode that will replay the batch currently being described."""

    NONE = "NONE"
    PIECEWISE = "PIECEWISE"
    FULL = "FULL"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_cudagraph_mode(cls, cudagraph_mode: Any) -> CPGraphMode:
        """Normalize a ``CUDAGraphMode`` (or a look-alike) into ``CPGraphMode``.

        ``FULL_AND_PIECEWISE`` reports both kinds of graphs. Without a batch
        level hint the piecewise interpretation wins, because that is the mode
        used for mixed and prefill batches; decode-only batches override it
        through ``full_graph=True`` in ``CPGraphSpec.from_runtime``.
        """
        if cudagraph_mode is None:
            return cls.NONE

        has_piecewise = getattr(cudagraph_mode, "has_piecewise_cudagraphs", None)
        if callable(has_piecewise) and has_piecewise():
            return cls.PIECEWISE

        has_full = getattr(cudagraph_mode, "has_full_cudagraphs", None)
        if callable(has_full) and has_full():
            return cls.FULL

        name = getattr(cudagraph_mode, "name", None) or str(cudagraph_mode)
        name = str(name).upper()
        if "PIECEWISE" in name:
            return cls.PIECEWISE
        if "FULL" in name:
            return cls.FULL
        return cls.NONE


@dataclass(frozen=True)
class CPGraphSpec:
    """How the current CP batch is going to be executed.

    ``padded_num_tokens`` is the dispatcher's token extent for this batch and is
    identical on every rank of a CP group. ``padded_num_reqs`` is only populated
    for full graphs: piecewise graphs pad tokens but never requests.
    """

    mode: CPGraphMode = CPGraphMode.NONE
    padded_num_tokens: int = 0
    padded_num_reqs: int | None = None
    capturing: bool = False

    def __post_init__(self) -> None:
        if self.padded_num_tokens < 0:
            raise ValueError(f"padded_num_tokens must be >= 0, got {self.padded_num_tokens}")
        if self.mode is not CPGraphMode.FULL and self.padded_num_reqs is not None:
            raise ValueError(
                f"Request padding only exists for full graphs; mode={self.mode}, padded_num_reqs={self.padded_num_reqs}"
            )

    @property
    def is_none(self) -> bool:
        return self.mode is CPGraphMode.NONE

    @property
    def is_piecewise(self) -> bool:
        return self.mode is CPGraphMode.PIECEWISE

    @property
    def is_full(self) -> bool:
        return self.mode is CPGraphMode.FULL

    @property
    def pads_requests(self) -> bool:
        """Whether request-shaped metadata is padded to the graph extent."""
        return self.is_full and self.padded_num_reqs is not None

    @property
    def uses_padded_token_extent(self) -> bool:
        """Whether padded rows beyond the scheduled tokens are part of the batch."""
        return not self.is_none and self.padded_num_tokens > 0

    @property
    def needs_graph_param_update(self) -> bool:
        """Whether attention kernel arguments must be patched before replay.

        Only full graphs capture the attention kernel itself; piecewise graphs
        run the attention op eagerly and read the current metadata directly.
        """
        return self.is_full

    def as_capture(self) -> CPGraphSpec:
        """Same layout, built for smoke capture or graph capture."""
        return replace(self, capturing=True)

    @classmethod
    def from_runtime(
        cls,
        cudagraph_mode: Any,
        *,
        full_graph: bool = False,
        padded_num_tokens: int = 0,
        padded_num_reqs: int | None = None,
        capturing: bool = False,
    ) -> CPGraphSpec:
        """Resolve the spec for the batch a caller is about to describe.

        ``full_graph`` is the batch-level hint computed by the runner (the
        legacy ``full_graph_mode`` argument) and is authoritative: capture paths
        and decode-only routines set it, mixed and prefill batches do not.
        ``cudagraph_mode`` may be the configuration mode or the runtime mode;
        a configuration that merely *offers* full graphs must not turn a
        non-decode batch into a full-graph batch.
        """
        resolved = CPGraphMode.from_cudagraph_mode(cudagraph_mode)
        if full_graph:
            mode = CPGraphMode.FULL
        elif resolved is CPGraphMode.PIECEWISE:
            mode = CPGraphMode.PIECEWISE
        elif resolved is CPGraphMode.FULL:
            # Full graphs only replay decode-only routines; a prefill or mixed
            # batch on the same configuration runs eagerly.
            mode = CPGraphMode.NONE
        else:
            mode = CPGraphMode.NONE
        if mode is not CPGraphMode.FULL:
            padded_num_reqs = None
        return cls(
            mode=mode,
            padded_num_tokens=padded_num_tokens,
            padded_num_reqs=padded_num_reqs,
            capturing=capturing,
        )


@dataclass(frozen=True)
class CPCapability:
    """Startup-time answer for "may this backend run CP with graphs?"."""

    supported: bool
    reason: str = ""

    @classmethod
    def yes(cls) -> CPCapability:
        return cls(supported=True)

    @classmethod
    def no(cls, reason: str) -> CPCapability:
        return cls(supported=False, reason=reason)

    def raise_if_unsupported(self) -> None:
        if not self.supported:
            raise NotImplementedError(self.reason)


_PIECEWISE_READY_FAMILIES = frozenset({"mla", "sfa", "gqa"})


def cp_piecewise_capability(
    *,
    attention_family: str,
    pcp_size: int,
    dcp_size: int,
    cudagraph_mode: Any = None,
) -> CPCapability:
    """Decide whether a CP configuration can run with piecewise graphs.

    Pure function so it can be unit tested without vLLM or NPU hardware.
    """
    mode = CPGraphMode.from_cudagraph_mode(cudagraph_mode)
    if mode is CPGraphMode.NONE:
        return CPCapability.yes()
    if attention_family in _PIECEWISE_READY_FAMILIES:
        return CPCapability.yes()
    if attention_family == DSA_FAMILY:
        if dcp_size > 1:
            return CPCapability.no(
                "DSA context parallelism is PCP-only; set decode_context_parallel_size=1 or disable graph mode."
            )
        return CPCapability.yes()
    return CPCapability.no(
        f"Context-parallel piecewise graphs do not support attention family '{attention_family}' "
        f"(PCP={pcp_size}, DCP={dcp_size}); use eager mode or FULL_DECODE_ONLY."
    )


def attention_family(vllm_config: Any) -> str:
    """Classify the model the same way the attention backend table does.

    Mirrors the ``(use_mla, use_sparse, use_compress)`` key used by
    ``NPUPlatform._select_attn_backend_cls`` without importing torch.
    """
    model_config = getattr(vllm_config, "model_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is not None:
        # DSA keeps a compressed cache; SFA only adds the sparse indexer.
        if hasattr(hf_text_config, "compress_ratios"):
            return DSA_FAMILY
        if hasattr(hf_text_config, "index_topk"):
            return "sfa"
    if getattr(model_config, "use_mla", False):
        return "mla"
    return "gqa"


def cp_piecewise_capability_for_config(vllm_config: Any) -> CPCapability:
    """``cp_piecewise_capability`` bound to a resolved ``VllmConfig``."""
    parallel_config = vllm_config.parallel_config
    compilation_config = vllm_config.compilation_config
    return cp_piecewise_capability(
        attention_family=attention_family(vllm_config),
        pcp_size=parallel_config.prefill_context_parallel_size,
        dcp_size=parallel_config.decode_context_parallel_size,
        cudagraph_mode=compilation_config.cudagraph_mode,
    )
