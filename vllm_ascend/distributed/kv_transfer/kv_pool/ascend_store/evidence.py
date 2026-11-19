# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-attempt KV transfer evidence for the AscendStore blockwise worker.

The production-side evidence producer of the hole-repair policy
(kv_load_failure_policy="repair"). One AscendStoreAttemptEvidence instance
is created at the START of every external-KV load attempt (sync or async),
carries the scheduler-assigned transfer_generation captured once from the
load spec, and accumulates the ACTUAL outcomes of every store GET the
attempt issued. Nothing here invents success: an attempt that did not
complete never reports, and an attempt whose failure geometry does not fit
the v1 single-region contract emits no report at all (the scheduler's
evidence gates then fail the delivery closed to recomputation).

Accounting unit
---------------
A blockwise AscendStore object is one (TP rank, KV-cache group, logical
block) chunk whose payload packs EVERY KV-cache layer of its group for that
block: the load enumeration emits exactly one store key per logical block
per participating group (ChunkedTokenDatabase._iter_token_chunks groups the
fine-grained chained hashes up to the group's block size), and prepare_value
spans every cache entry (layer) of the group. The scheduler's evidence
contract reconciles reports and attestations in (block, layer) cells --
failed_objects == len(failed_block_ids) * len(observed_failed_layers) per
rank -- so this module counts in the same cell unit:

* expected_cells -- blocks this rank expected to load, times the layer
  count.
* failed_cells -- failed blocks times the layer count.
* certified_cells -- blocks fully loaded under the claimed certification
  mode, times the layer count.

Layer-level failure precision does not exist on this path: a GET failure
poisons the WHOLE block across every layer, so an honest report names every
layer. The layer count is the TARGET model's attention-layer count
(hf_config.num_hidden_layers) -- the exact bound the scheduler's report
battery enforces -- NOT the worker's num_layers, which additionally counts
EAGLE/MTP draft layers and would make every honest report read as
malformed.

Truthfulness rules
------------------
* The generation is captured once at attempt start and echoed unchanged in
  both the report and the attestation; it is never re-read at emission time,
  so a delayed old attempt can never inherit a newer attempt's identity.
* Certification is only ever a mode the GET path actually applies:
  "content" -- the blockwise store keys embed the block's chained token hash
  (an identity key, NOT a digest of the KV payload bytes) and the mooncake
  store's object model gives the deployment the documented store-trust
  assumption (atomic per-object writes; GET returns the bytes bound to the
  identity key), mirroring the upstream mooncake producer's claim for the
  same store and key scheme. "none" -- no verification beyond GET success.
  The memcache and yuanrong backends are FORCED to "none": the only
  write-once property available to them is a CLIENT-SIDE exists-filter
  (lookup, then put the missing -- a racy check-then-write, not a
  service-side add-if-absent), yuanrong additionally re-PUTs every key on
  every save, and memcache's blockwise put path exposes no
  atomic-visibility contract (its write-finish handshake exists only in
  the layerwise plane). The key derivation covers tokens + model +
  parallel coordinates but NOT the full payload determinants (dtype,
  quantization, kernel numerics), so same-key-different-bytes cannot be
  ruled out across deployments. Digest verification ("sha256") is never
  claimed: no AscendStore GET path performs an independent digest
  check.
* An attempt that did not complete (interrupted, exception before
  terminality) never reports success: it stays non-terminal, its attestation
  is not emitted, and the scheduler's completeness gate fails the delivery
  closed to recomputation.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

# The v1 evidence contract classes. Imported defensively: vllm-ascend must
# stay importable against pinned vLLM cores that predate the PR1-3 contract
# (there the evidence path is fully disabled and the flat failure channels
# behave exactly as before).
try:  # pragma: no cover - trivial import guard
    from vllm.v1.outputs import KVLoadAttestation, KVLoadFailureReport

    _CONTRACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    KVLoadAttestation = None  # type: ignore[assignment]
    KVLoadFailureReport = None  # type: ignore[assignment]
    _CONTRACT_AVAILABLE = False

EVIDENCE_CERTIFICATION_MODES = ("content", "none")


def contract_available() -> bool:
    """Whether the vLLM core ships the v1 evidence contract classes."""
    return _CONTRACT_AVAILABLE


def validate_evidence_certification(mode: str) -> str:
    """Validate the configured evidence certification mode.

    Only "content" (content-addressed blockwise store keys under the
    documented store-trust assumption) and "none" (no verification beyond
    GET success) are producible by this worker: the GET path performs no
    independent digest check, so "sha256" is rejected -- never claim a
    verification that did not happen.
    """
    if mode not in EVIDENCE_CERTIFICATION_MODES:
        raise ValueError(
            "AscendStore evidence_certification must be "
            '"content" or "none" (got ' + repr(mode) + "); the sha256 mode "
            "requires manifest support that the store GET path does "
            "not provide"
        )
    return mode


def evidence_enabled(
    kv_load_failure_policy,
    *,
    contract_importable: bool,
    num_prefix_cacheable_groups: int,
    num_kv_cache_groups: int,
    use_hybrid: bool,
    use_layerwise: bool,
    tp_mismatch: bool = False,
    use_kvpp: bool = False,
    dcp_size: int = 1,
    pcp_size: int = 1,
    pp_size: int = 1,
    group_uses_align_state: bool = False,
) -> bool:
    """Whether the structured evidence channels may activate.

    Activation gates (all must hold):

    * the pinned vLLM core must actually ship the evidence contract -- a
      pre-PR1 core cannot consume the reports, so emitting them would only
      crash its (absent) evidence handler while the flat channels carry the
      failure;
    * the repair policy must be explicitly configured -- under any other
      policy (including the default "fail") the connector's evidence output
      would raise in the scheduler's evidence handler and destroy the only
      working failure channel, so the evidence channels stay off;
    * exactly ONE prefix-cacheable group in a single-group (non-hybrid,
      non-align-state) layout: store block ids are group-local, so
      evidence from a multi-group scheduler cannot be represented without
      conflation, and the scheduler's report battery is single-group-shaped
      (it rejects multi-group at init);
    * the blockwise transfer structure only: the layerwise planes have a
      different per-layer key granularity and per-plane readiness
      protocols, the TP-mismatch path shards one block across multiple
      sub-keys, and KVPP shards layers across ranks -- none of these can
      honestly produce single-group blockwise evidence;
    * PP=1, CP=1, DCP=1: the scheduler's completeness gate is TP-scoped and
      its report battery rejects everything else at init.
    """
    if not contract_importable:
        return False
    if kv_load_failure_policy != "repair":
        return False
    if use_layerwise or tp_mismatch or use_kvpp:
        return False
    if use_hybrid or group_uses_align_state:
        return False
    if num_prefix_cacheable_groups != 1 or num_kv_cache_groups != 1:
        return False
    return dcp_size == 1 and pcp_size == 1 and pp_size == 1


@dataclass
class AscendStoreAttemptEvidence:
    """Mutable per-attempt bookkeeping owned by one load path.

    Thread-safety: a single attempt is owned by exactly one execution context
    (the worker's main thread for sync loads, the receiving thread for async
    loads); the hand-off buffer below is the only shared structure.
    """

    req_id: str
    transfer_generation: int | None
    # The TARGET model's attention-layer count (hf_config.num_hidden_layers),
    # NOT the worker's num_layers (which counts EAGLE/MTP draft layers too).
    num_layers: int
    block_size: int
    # The attempt's restoration target: the external claim's absolute token
    # extent (LoadSpec.kvpool_cached_tokens).
    restored_extent: int
    certification_mode: str
    # Logical (request-absolute) block indices this rank expected to load.
    expected_blocks: set = field(default_factory=set)
    # Blocks with at least one failed store GET (or an exception mid-batch).
    failed_blocks: set = field(default_factory=set)
    # Blocks whose GET acknowledged success.
    loaded_blocks: set = field(default_factory=set)
    # Logical index -> physical block id (the scheduler's block table for
    # this request), captured during key enumeration.
    block_ids: dict = field(default_factory=dict)
    terminal: bool = False

    def note_expected_block(self, logical_block: int) -> None:
        self.expected_blocks.add(logical_block)

    def note_block_outcome(self, logical_block: int, ok: bool) -> None:
        """Record the outcome of the store object covering logical_block.

        A block counts as loaded only when its GET succeeded; any failure
        poisons the whole block across every layer.
        """
        if ok:
            self.loaded_blocks.add(logical_block)
        else:
            self.failed_blocks.add(logical_block)
            self.loaded_blocks.discard(logical_block)

    def note_blocks_failed(self, logical_blocks: Iterable) -> None:
        for block in logical_blocks:
            self.note_block_outcome(block, False)

    def finalize(self) -> None:
        """Mark the attempt terminal. Called exactly once, when the load
        handler for the request completes on any path (success, per-key GET
        failure, whole-batch failure, or an empty key list)."""
        self.terminal = True

    # ---- derived accounting (cell unit: (block, layer)) ----

    @property
    def expected_cells(self) -> int:
        return len(self.expected_blocks) * self.num_layers

    @property
    def loaded_cells(self) -> int:
        return len(self.loaded_blocks - self.failed_blocks) * self.num_layers

    @property
    def failed_cells(self) -> int:
        return len(self.failed_blocks) * self.num_layers

    @property
    def certified_cells(self) -> int:
        # The certification covers exactly the LOADED set (the failed
        # objects are the report's region -- excluded by construction:
        # certifies_content() on the attestation checks certified ==
        # loaded). Only "content" is producible here: "sha256" requires
        # per-object digest verification the store GET path does not
        # perform (validate_evidence_certification rejects it at
        # configuration time), so the accounting refuses to ever count it
        # as certified -- even a directly-constructed attempt cannot claim
        # a verification that did not happen.
        if self.certification_mode == "content":
            return self.loaded_cells
        return 0

    @property
    def certifies_content(self) -> bool:
        """Whether the loaded set is fully certified under the mode (mirrors
        KVLoadAttestation.certifies_content). Only "content" can certify
        here -- see certified_cells."""
        if self.certification_mode != "content":
            return False
        return self.certified_cells > 0 and self.certified_cells == self.loaded_cells

    def to_attestation(self):
        if KVLoadAttestation is None:  # pragma: no cover - gate prevents this
            raise RuntimeError("the v1 evidence contract is not importable; the evidence path must not be active")
        return KVLoadAttestation(
            request_id=self.req_id,
            expected_objects=self.expected_cells,
            loaded_objects=self.loaded_cells,
            failed_objects=self.failed_cells,
            certification=self.certification_mode,
            certified_objects=self.certified_cells,
            certification_failures=0,
            terminal=self.terminal,
            transfer_generation=self.transfer_generation,
        )

    def to_failure_report(self):
        """Build the v1 single-region failure report for this attempt.

        Returns None when the actual failure geometry cannot be represented
        by the single-region contract, and the caller emits no report: the
        scheduler's evidence gates then fail the delivery closed to
        recomputation -- never an invented repairable region.
        Unrepresentable shapes:

        * more than one failed block (v1 supports exactly one region);
        * a failed block without a captured physical id;
        * a failed block whose block-aligned span would exceed the
          attempt's restoration target (the unaligned final block: the
          battery requires span end <= restored_extent and both
          block-aligned, and padding the extent would claim content the
          attempt never vouched for).

        The rollback boundary is the block-aligned start of the failed
        block iff EVERY preceding expected block of the attempt loaded; any
        hole below the failure (a masked-out, never-enumerated, or failed
        block) forces the full-recompute floor (rollback 0). Blocks below
        the attempt's delta start are the request's LOCAL prefix, not
        external content, so they need no transfer evidence.
        """
        if KVLoadFailureReport is None:  # pragma: no cover - gate prevents this
            raise RuntimeError("the v1 evidence contract is not importable; the evidence path must not be active")
        if len(self.failed_blocks) != 1:
            return None
        logical = next(iter(self.failed_blocks))
        block_id = self.block_ids.get(logical)
        if block_id is None:
            return None
        span_start = logical * self.block_size
        span_end = span_start + self.block_size
        if span_end > self.restored_extent:
            return None
        rollback = span_start
        for idx in sorted(self.expected_blocks):
            if idx >= logical:
                break
            if idx in self.failed_blocks or idx not in self.loaded_blocks:
                rollback = 0
                break
        return KVLoadFailureReport(
            request_id=self.req_id,
            failed_block_ids=[block_id],
            token_span=(span_start, span_end),
            rollback_boundary=rollback,
            observed_failed_layers=list(range(self.num_layers)),
            healthy_above_span=self.certifies_content,
            restored_extent=self.restored_extent,
            transfer_generation=self.transfer_generation,
        )


class EvidenceBuffer:
    """Thread-safe hand-off of completed attempt evidence from the load
    execution contexts (worker main thread for sync loads, the receiving
    thread for async loads) to the worker's get_load_* drain (called once
    per step in post_forward)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts: dict = {}
        # Terminal evidence displaced by a newer in-flight attempt (kept
        # for the next drain instead of being destroyed).
        self._drained_ready: list = []

    def start_attempt(self, evidence: AscendStoreAttemptEvidence) -> None:
        """Register a new attempt, replacing any stale entry for the same
        request, EXCEPT when the registered entry must win:

        * it carries a strictly NEWER generation (a delayed old attempt
          registering after a new attempt's must never clobber the current
          attempt's evidence); or
        * it is IDENTIFIED while the new one is identity-less (an
          identity-less attempt can never produce admissible evidence --
          the scheduler's partition rejects it -- so replacing an
          identified attempt with it would only destroy the request's
          potentially valid evidence).
        """
        with self._lock:
            existing = self._attempts.get(evidence.req_id)
            if existing is not None:
                if (
                    existing.transfer_generation is not None
                    and evidence.transfer_generation is not None
                    and evidence.transfer_generation < existing.transfer_generation
                ):
                    return
                if existing.transfer_generation is not None and evidence.transfer_generation is None:
                    return
                if existing.terminal and not evidence.terminal:
                    # A TERMINAL-but-undrained entry's evidence is complete
                    # and about to be consumed by this step's drain; a newer
                    # NON-terminal registration (an in-flight attempt that
                    # just started) must not destroy it -- the delivery
                    # would silently demote to a full recompute. Move the
                    # terminal evidence to the drain-ready list and register
                    # the new attempt: BOTH survive, each in its own channel.
                    self._drained_ready.append(existing)
                    self._attempts[evidence.req_id] = evidence
                    return
            self._attempts[evidence.req_id] = evidence

    def get(self, req_id: str) -> AscendStoreAttemptEvidence | None:
        with self._lock:
            return self._attempts.get(req_id)

    def drain_terminal(self) -> list:
        """Remove and return every terminal attempt. Non-terminal entries
        stay for a later pass (their loads are still in flight)."""
        with self._lock:
            terminal = [ev for ev in self._attempts.values() if ev.terminal]
            for ev in terminal:
                del self._attempts[ev.req_id]
            if self._drained_ready:
                terminal.extend(self._drained_ready)
                self._drained_ready = []
            return terminal

    def drop(self, req_id: str) -> None:
        """Forget a request pending evidence (finished/preempted without
        terminality, connector shutdown, or a crashed attempt handler)."""
        with self._lock:
            self._attempts.pop(req_id, None)

    def drop_if_identity(
        self,
        req_id: str,
        transfer_generation,
    ) -> None:
        """Drop ONLY the entry whose identity matches transfer_generation.

        Used by a crashed attempt handler: an identity-blind drop could
        destroy a NEWER attempt's registered evidence (an availability
        loss -- the delivery would fail closed with the terminal
        attestation lost). An identity-less crash can only drop an
        identity-less entry: a crashed no-identity handler must never
        destroy a REGISTERED attempt's evidence.
        """
        with self._lock:
            current = self._attempts.get(req_id)
            if current is None:
                return
            if current.transfer_generation == transfer_generation:
                del self._attempts[req_id]

    def get_attempt(self, req_id: str) -> AscendStoreAttemptEvidence | None:
        """Return the request's currently registered attempt (read-only
        peek for identity-aware drop decisions; never mutates state)."""
        with self._lock:
            return self._attempts.get(req_id)
