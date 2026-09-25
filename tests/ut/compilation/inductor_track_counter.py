"""In-process observability counters for the inductor track (stage4 W4, U3).

Relocated 2026-09-24 from ``vllm_ascend/compilation/`` to ``tests/`` per the
stage4 debt-8 ruling (test-support code must not live in product sources);
the module had zero product consumers at migration time.

stage design/stage4/02_设计方案.md §四-4 promised an in-process
``AscendCompilationCounter`` (U3 "治理＋可观测全量收官"): the offline reporter
(``docs/vllm/env/scripts/inductor_cache_stats.py``) scans compile artifacts
from a shell; this module exposes the same D5 tri-state accounting as a
library API any in-process test or probe can call once the engine is up.

D5 tri-state (authoritative third state = the runtime-enumerated
``torch_npu._inductor.triton_experimental.lowering.FALLBACK_LIST``):

* ``fallback_listed`` -> ``num_aclnn_fallbacks``  (expected aclnn extern fallback)
* ``opaque_custom``   -> ``num_opaque_custom``    (vllm/_C_ascend opaque ops, extern by design)
* ``unclassified``    -> ``num_unclassified``     (no TE lowering & not listed — coverage gap)

Usage (in-process, after the compiled engine is up)::

    from tests.ut.compilation.inductor_track_counter import (
        AscendCompilationCounter,
    )

    counter = AscendCompilationCounter.capture()
    counter.expect(num_unclassified=0)             # absolute assertion
    counter.expect_at_least(num_aclnn_fallbacks=1)  # threshold assertion
"""

import dataclasses
import glob
import os
import re
from collections.abc import Iterator

# Same criteria as the offline reporter (inductor_cache_stats.py):
# triton_experimental codegen marker / TE kernel decorator / extern call sites.
EXPERIMENTAL_MARKER = "triton_experimental import npu_triton_heuristics"
TE_KERNEL_RE = re.compile(r"@npu_triton_heuristics\.(\w+)\(")
CALL_RE = re.compile(r"\b(extern_kernels\.[A-Za-z_][\w.]*|torch\.ops\.[A-Za-z_][\w.]*)\(")
STRING_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
# Non-operator namespaces (memory-pool infra) — not extern accounting.
INFRA_NAMESPACES = {"inductor"}


def _load_fallback_authority() -> set[str] | None:
    """Enumerate lowering.FALLBACK_LIST after activating triton_experimental
    (CPU-side, T0a-2). Returns None when the authority cannot be loaded —
    affected sites then land in ``num_authority_unavailable`` instead of being
    silently guessed."""
    try:
        import torch_npu  # noqa: F401
        import torch_npu._inductor as tni

        tni._load_triton_experimental_backend()
        from torch_npu._inductor.triton_experimental import lowering

        return {str(op) for op in lowering.FALLBACK_LIST}
    except Exception:  # noqa: BLE001 - observability must never break the caller
        return None


def _resolve_op(qual: str):
    """'aten.embedding' -> the torch.ops object, or None when unresolvable."""
    import torch

    parts = qual.split(".")
    for keep in (len(parts), len(parts) - 1):  # full name, else drop one segment
        obj = torch.ops
        try:
            for part in parts[:keep]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def _classify(op: str, authority: set[str] | None) -> str | None:
    """D5 tri-state for one extern op name; None = infra namespace (skip)."""
    ns = op.split(".", 1)[0]
    if ns in INFRA_NAMESPACES:
        return None
    if ns in ("vllm", "_C_ascend"):
        return "opaque_custom"
    if authority is None:
        return "authority_unavailable"
    resolved = _resolve_op(op)
    if resolved is None:
        # aten/npu core namespaces failing to resolve is a naming anomaly —
        # register as unclassified rather than silently dropping.
        return "unclassified" if ns in ("aten", "npu") else "opaque_custom"
    name = str(resolved)
    if name in authority or name + ".default" in authority:
        return "fallback_listed"
    return "unclassified"


def _iter_wrapper_texts(cache_root: str) -> Iterator[str]:
    """Yield output_code.py wrapper texts (compile pieces) under a cache root."""
    for rank_dir in sorted(glob.glob(os.path.join(cache_root, "**", "rank_*"), recursive=True)):
        for path in sorted(
            glob.glob(os.path.join(rank_dir, "inductor_cache", "*", "*.py"))
        ):
            if os.sep + ".debug" + os.sep in path:
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            if text.startswith("# AOT ID:"):
                yield text


@dataclasses.dataclass
class AscendCompilationCounter:
    """Tri-state extern accounting for the inductor track (D5 口径).

    Artifact-derived fields (any process that can read the cache directory):
    ``num_triton_kernels`` counts TE-decorated kernels across wrappers; the
    extern fields count classified call sites. ``num_generated_kernels_metrics``
    reads torch inductor metrics and is only meaningful in-process, after
    compiles ran in this interpreter.
    """

    num_triton_kernels: int = 0
    num_aclnn_fallbacks: int = 0
    num_opaque_custom: int = 0
    num_unclassified: int = 0
    num_authority_unavailable: int = 0
    num_wrappers_scanned: int = 0
    num_generated_kernels_metrics: int = 0

    @classmethod
    def capture(
        cls,
        cache_root: str | os.PathLike | None = None,
        *,
        read_metrics: bool = True,
        fallback_authority: set[str] | None = None,
    ) -> "AscendCompilationCounter":
        """Snapshot the counters (artifact scan + optional torch metrics).

        ``fallback_authority`` defaults to the runtime-enumerated FALLBACK_LIST
        and can be injected for deterministic tests.
        """
        if cache_root is None:
            cache_root = os.environ.get(
                "VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm")
            )
        if fallback_authority is None and os.path.isdir(cache_root):
            fallback_authority = _load_fallback_authority()

        counter = cls()
        if read_metrics:
            try:
                import torch._inductor.metrics as metrics

                counter.num_generated_kernels_metrics = int(
                    getattr(metrics, "generated_kernel_count", 0)
                )
            except Exception:  # noqa: BLE001 - metrics are best-effort
                pass
        if not os.path.isdir(cache_root):
            return counter

        for text in _iter_wrapper_texts(str(cache_root)):
            counter.num_wrappers_scanned += 1
            if EXPERIMENTAL_MARKER not in text:
                continue  # default-codegen piece — no TE accounting
            counter.num_triton_kernels += len(TE_KERNEL_RE.findall(text))
            for line in text.splitlines():
                code = STRING_RE.sub("", line.split("#", 1)[0])
                for m in CALL_RE.finditer(code):
                    call = m.group(1)
                    if call.startswith("extern_kernels."):
                        op = "aten." + call.split(".", 1)[1].split("(")[0]
                    else:
                        parts = call[len("torch.ops."):].rstrip("(").split(".")
                        if parts[-1] in ("default", "out"):
                            parts = parts[:-1]
                        op = ".".join(parts)
                    state = _classify(op, fallback_authority)
                    if state == "fallback_listed":
                        counter.num_aclnn_fallbacks += 1
                    elif state == "opaque_custom":
                        counter.num_opaque_custom += 1
                    elif state == "unclassified":
                        counter.num_unclassified += 1
                    elif state == "authority_unavailable":
                        counter.num_authority_unavailable += 1
        return counter

    def expect(self, **kwargs: int) -> None:
        """Assert absolute values (upstream compilation_counter.expect parity)."""
        for key, value in kwargs.items():
            actual = getattr(self, key)
            assert actual == value, (
                f"{key} not as expected: expected {value}, got {actual} ({self})"
            )

    def expect_at_least(self, **kwargs: int) -> None:
        """Assert threshold values (gate form: proves the scan found artifacts)."""
        for key, value in kwargs.items():
            actual = getattr(self, key)
            assert actual >= value, (
                f"{key} below threshold: expected >= {value}, got {actual} ({self})"
            )
