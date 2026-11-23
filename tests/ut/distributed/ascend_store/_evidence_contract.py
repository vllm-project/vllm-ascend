"""Install the PR1-3 evidence contract classes for UT runs.

The ascend_store UT suite runs against a mocked vllm (see _mock_deps):
there vllm.v1.outputs is a MagicMock, so importing the real dataclasses
is impossible. This helper injects VERBATIM-semantics stand-ins of the
v1 KVLoadFailureReport / KVLoadAttestation contracts (field order and the
well_formed / certifies_content validation battery, copied from the
PR1-3 vllm/v1/outputs.py) into the mocked module BEFORE the evidence
module is imported, so the accounting arithmetic the tests pin matches
the real scheduler battery exactly.

Under a real PR1-3 core (e.g. PYTHONPATH-shadowed CI runs) the classes
are already importable and identical; install() then keeps them.
"""

import sys
from dataclasses import dataclass


@dataclass(slots=True)
class KVLoadFailureReport:
    request_id: str
    failed_block_ids: list
    token_span: tuple
    rollback_boundary: int
    observed_failed_layers: list
    healthy_above_span: bool
    restored_extent: int
    source_rank: int | None = None
    transfer_generation: int | None = None


@dataclass(slots=True)
class KVLoadAttestation:
    request_id: str
    expected_objects: int
    loaded_objects: int
    failed_objects: int
    certification: str
    certified_objects: int
    certification_failures: int
    terminal: bool
    transfer_generation: int | None = None
    source_rank: int | None = None

    def well_formed(self) -> bool:
        if (
            self.expected_objects < 0
            or self.loaded_objects < 0
            or self.failed_objects < 0
            or self.certified_objects < 0
            or self.certification_failures < 0
        ):
            return False
        if self.certification not in ("sha256", "content", "size", "none"):
            return False
        if self.terminal and (self.loaded_objects + self.failed_objects != self.expected_objects):
            return False
        if self.certified_objects + self.certification_failures > self.loaded_objects:
            return False
        if self.certification_failures > 0:
            return self.certification in ("sha256", "content")
        return True

    def certifies_content(self) -> bool:
        if self.certification not in ("sha256", "content"):
            return False
        if self.expected_objects < 1:
            return False
        return self.certification_failures == 0 and self.certified_objects == self.loaded_objects


def install() -> None:
    """Put the contract classes into vllm.v1.outputs if (and only if) the
    module lacks the real ones. Idempotent."""
    outputs = sys.modules.get("vllm.v1.outputs")
    if outputs is None:
        return
    existing = getattr(outputs, "KVLoadAttestation", None)
    if isinstance(existing, type) and hasattr(existing, "certifies_content"):
        # Real (or already-installed) classes: keep them.
        return
    outputs.KVLoadAttestation = KVLoadAttestation  # type: ignore[attr-defined]
    outputs.KVLoadFailureReport = KVLoadFailureReport  # type: ignore[attr-defined]
