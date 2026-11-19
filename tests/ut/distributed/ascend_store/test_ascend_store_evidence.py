#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Structured KV recovery evidence producer for AscendStore (repair policy).

The producer contract: real blockwise store-GET outcomes are mapped, per
attempt and per transfer generation, into the structured failure reports
and terminal per-rank attestations the v1 scheduler's evidence gates
consume (kv_load_failure_policy="repair"). Drives the REAL blockwise load
paths (KVPoolWorker.start_load_kv sync and KVCacheStoreRecvingThread async)
with realistic per-key AscendStore result semantics (mooncake: int >= 0
success, < 0 failure e.g. OBJECT_NOT_FOUND -704, FILE_READ_FAIL -1102;
memcache: 0 ok / !=0 fail; yuanrong: 0 ok / 1 fail; None = the backend
caught an exception => whole batch failed), the real EvidenceBuffer
lifecycle, and the real worker getters.

Scheduler-distinguishable states pinned by these tests:
(1) fully healthy transfer; (2) completed with known failures;
(3) incomplete; (4) missing rank evidence; (5) stale results.
Each test names the state(s) it pins.
"""

import threading
import unittest
from unittest.mock import MagicMock

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from tests.ut.distributed.ascend_store.cache_config_helpers import (  # noqa: E402
    make_resolved_cache_config,
)

# The PR1-3 evidence contract classes. Under the mocked-vllm UT
# environment the real dataclasses are injected by the helper below with
# the VERBATIM v1 semantics (well_formed / certifies_content), so the
# accounting arithmetic this suite pins matches the real battery. Under a
# real PR1-3 core the classes are already present and identical.
import tests.ut.distributed.ascend_store._evidence_contract as _contract  # noqa: E402

_contract.install()

# isort: on
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.evidence import (  # noqa: E402
    AscendStoreAttemptEvidence,
    EvidenceBuffer,
    contract_available,
    evidence_enabled,
    validate_evidence_certification,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (  # noqa: E402
    KVCacheStoreRecvingThread,
    record_failed_blocks,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (  # noqa: E402
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (  # noqa: E402
    KVPoolWorker,
)

BLOCK = 16
LAYERS = 4
N_BLOCKS = 4
TOKEN_LEN = N_BLOCKS * BLOCK

# Realistic per-key AscendStore result codes.
RC_OBJECT_NOT_FOUND = -704  # mooncake: key absent (missing object)
RC_FILE_READ_FAIL = -1102  # mooncake: EIO-class disk read failure


class RcStore:
    """Returns caller-configured per-key rc lists from the batch GET,
    one consumed list per call (realistic per-key semantics). None is
    returned when the configured entry is None (backend exception).
    """

    def __init__(self, rcs):
        self.rcs = list(rcs)
        self.get_calls = []

    def set_device(self):
        pass

    def register_buffer(self, ptrs, lengths):
        pass

    def get(self, keys, addrs, sizes):
        self.get_calls.append(list(keys))
        out = self.rcs[: len(keys)]
        self.rcs = self.rcs[len(keys) :]
        if out and out[0] is None:
            return None
        return out


class ExplodingStore:
    def set_device(self):
        pass

    def register_buffer(self, ptrs, lengths):
        pass

    def get(self, keys, addrs, sizes):
        raise RuntimeError("store exploded")


def make_attempt(req_id="r1", gen=1, layers=LAYERS, block_size=BLOCK, extent=TOKEN_LEN, mode="content"):
    return AscendStoreAttemptEvidence(
        req_id=req_id,
        transfer_generation=gen,
        num_layers=layers,
        block_size=block_size,
        restored_extent=extent,
        certification_mode=mode,
    )


def fill_attempt(ev, n=N_BLOCKS, failed=()):
    """Mark n expected blocks (physical ids 100+idx), fail the given
    logical indices, load the rest."""
    for i in range(n):
        ev.note_expected_block(i)
        ev.block_ids[i] = 100 + i
        ev.note_block_outcome(i, i not in failed)
    return ev


class TestAttemptEvidenceAccounting(unittest.TestCase):
    """Cell accounting, report geometry, certification honesty."""

    def test_healthy_attempt_attestation_is_complete_and_certified(self):
        # [1] fully healthy transfer: expected == loaded, failed == 0.
        ev = fill_attempt(make_attempt())
        ev.finalize()
        att = ev.to_attestation()
        self.assertTrue(att.terminal)
        self.assertTrue(att.well_formed())
        self.assertTrue(att.certifies_content())
        self.assertEqual(att.expected_objects, N_BLOCKS * LAYERS)
        self.assertEqual(att.loaded_objects, N_BLOCKS * LAYERS)
        self.assertEqual(att.failed_objects, 0)
        self.assertEqual(att.certified_objects, att.loaded_objects)
        self.assertEqual(att.transfer_generation, 1)

    def test_missing_object_single_region_report(self):
        # [2] completed with known failures: exactly one failed block.
        ev = fill_attempt(make_attempt(), failed=(2,))
        ev.finalize()
        att = ev.to_attestation()
        rep = ev.to_failure_report()
        self.assertTrue(att.well_formed())
        # Reconciliation: failed == one block x all layers.
        self.assertEqual(att.failed_objects, 1 * LAYERS)
        self.assertEqual(att.loaded_objects, 3 * LAYERS)
        self.assertEqual(rep.failed_block_ids, [102])
        self.assertEqual(rep.token_span, (32, 48))
        self.assertEqual(rep.rollback_boundary, 32)  # contiguous prefix
        self.assertEqual(rep.observed_failed_layers, list(range(LAYERS)))
        self.assertEqual(rep.restored_extent, TOKEN_LEN)
        self.assertEqual(rep.transfer_generation, 1)
        self.assertTrue(rep.healthy_above_span)

    def test_hole_below_failure_forces_full_recompute_rollback(self):
        # [2] a never-loaded block below the failure is a hole: rollback 0.
        ev = make_attempt()
        for i in range(N_BLOCKS):
            ev.note_expected_block(i)
            ev.block_ids[i] = 100 + i
        ev.note_block_outcome(0, True)
        # block 1 never loaded (e.g. masked out): not in loaded set
        ev.note_block_outcome(2, False)
        ev.note_block_outcome(3, True)
        ev.finalize()
        rep = ev.to_failure_report()
        self.assertEqual(rep.rollback_boundary, 0)

    def test_multi_block_failure_not_representable(self):
        # [2] v1 supports exactly one region: no report, honest attestation.
        ev = fill_attempt(make_attempt(), failed=(0, 3))
        ev.finalize()
        self.assertIsNone(ev.to_failure_report())
        att = ev.to_attestation()
        self.assertEqual(att.failed_objects, 2 * LAYERS)
        self.assertTrue(att.well_formed())

    def test_unaligned_final_block_not_representable(self):
        # [2] restored_extent inside the failed block's span: padding would
        # claim content the attempt never vouched for.
        ev = fill_attempt(make_attempt(extent=50), failed=(3,))
        ev.finalize()
        self.assertIsNone(ev.to_failure_report())

    def test_non_terminal_attempt_never_reports(self):
        # [3] incomplete: no terminal attestation is emitted (stays buffered).
        ev = fill_attempt(make_attempt(), failed=(1,))
        att = ev.to_attestation()
        self.assertFalse(att.terminal)
        self.assertTrue(att.well_formed())

    def test_failed_block_without_physical_id_not_representable(self):
        # [4] a failed block Core cannot map to the request's table.
        ev = fill_attempt(make_attempt(), failed=(1,))
        del ev.block_ids[1]
        ev.finalize()
        self.assertIsNone(ev.to_failure_report())

    def test_certification_none_fails_closed(self):
        # [2] honest "we could not verify" never certifies content.
        ev = fill_attempt(make_attempt(mode="none"))
        ev.finalize()
        att = ev.to_attestation()
        self.assertEqual(att.certification, "none")
        self.assertEqual(att.certified_objects, 0)
        self.assertFalse(att.certifies_content())
        rep_shape = fill_attempt(make_attempt(mode="none"), failed=(1,))
        rep_shape.finalize()
        rep = rep_shape.to_failure_report()
        self.assertFalse(rep.healthy_above_span)

    def test_sha256_mode_rejected_at_configuration(self):
        with self.assertRaises(ValueError):
            validate_evidence_certification("sha256")
        self.assertEqual(validate_evidence_certification("none"), "none")
        self.assertEqual(validate_evidence_certification("content"), "content")

    def test_vacuous_attempt_never_certifies(self):
        # [4] expected set empty: no object to vouch for.
        ev = make_attempt()
        ev.finalize()
        att = ev.to_attestation()
        self.assertEqual(att.expected_objects, 0)
        self.assertFalse(att.certifies_content())


class TestEvidenceBufferGenerations(unittest.TestCase):
    """Generation isolation and cleanup."""

    def _buffered(self, gen, terminal=False):
        ev = make_attempt(gen=gen)
        ev.terminal = terminal
        return ev

    def test_stale_generation_registration_rejected(self):
        # [5] stale results: an old attempt's late registration cannot
        # clobber the current attempt's evidence.
        buf = EvidenceBuffer()
        buf.start_attempt(self._buffered(2))
        buf.start_attempt(self._buffered(1))
        self.assertEqual(buf.get_attempt("r1").transfer_generation, 2)

    def test_identityless_registration_cannot_replace_identified(self):
        buf = EvidenceBuffer()
        buf.start_attempt(self._buffered(3))
        buf.start_attempt(self._buffered(None))
        self.assertEqual(buf.get_attempt("r1").transfer_generation, 3)

    def test_terminal_evidence_survives_newer_inflight_attempt(self):
        # [1/5] drain both: the older terminal attempt's evidence is kept.
        buf = EvidenceBuffer()
        buf.start_attempt(self._buffered(1, terminal=True))
        buf.start_attempt(self._buffered(2))
        drained = buf.drain_terminal()
        self.assertEqual({e.transfer_generation for e in drained}, {1})
        self.assertEqual(buf.get_attempt("r1").transfer_generation, 2)

    def test_drop_releases_request(self):
        # [3] cancellation: preempted requests leave no stale evidence.
        buf = EvidenceBuffer()
        buf.start_attempt(self._buffered(1))
        buf.drop("r1")
        self.assertIsNone(buf.get_attempt("r1"))

    def test_drop_if_identity_is_identity_aware(self):
        # [4] a crashed handler drops only its own attempt's entry, never
        # a newer attempt's registered evidence.
        buf = EvidenceBuffer()
        buf.start_attempt(self._buffered(2))
        buf.drop_if_identity("r1", 1)  # old identity: no-op
        self.assertIsNotNone(buf.get_attempt("r1"))
        buf.drop_if_identity("r1", None)  # identity-less: no-op on 2
        self.assertIsNotNone(buf.get_attempt("r1"))
        buf.drop_if_identity("r1", 2)  # own identity: dropped
        self.assertIsNone(buf.get_attempt("r1"))

    def test_concurrent_attempts_kept_separate(self):
        # [1/2] concurrent requests cannot mix evidence.
        buf = EvidenceBuffer()
        ev_a = fill_attempt(make_attempt(req_id="r-a", gen=1))
        ev_b = fill_attempt(make_attempt(req_id="r-b", gen=2), failed=(0,))
        buf.start_attempt(ev_a)
        buf.start_attempt(ev_b)
        ev_a.finalize()
        ev_b.finalize()
        drained = buf.drain_terminal()
        by_req = {e.req_id: e for e in drained}
        self.assertEqual(by_req["r-a"].failed_cells, 0)
        self.assertEqual(by_req["r-b"].failed_cells, LAYERS)


class TestEvidenceGates(unittest.TestCase):
    """Activation gates: everything except the exact supported layout keeps
    the legacy flat channels (#9701) untouched."""

    def _ok_kwargs(self):
        return dict(
            contract_importable=True,
            num_prefix_cacheable_groups=1,
            num_kv_cache_groups=1,
            use_hybrid=False,
            use_layerwise=False,
        )

    def test_only_repair_policy_activates(self):
        kw = self._ok_kwargs()
        self.assertTrue(evidence_enabled("repair", **kw))
        for policy in ("fail", "recompute", None):
            self.assertFalse(evidence_enabled(policy, **kw))

    def test_unsupported_layouts_stay_off(self):
        kw = self._ok_kwargs()
        for overrides in (
            {"use_layerwise": True},
            {"tp_mismatch": True},
            {"use_kvpp": True},
            {"use_hybrid": True},
            {"group_uses_align_state": True},
            {"num_kv_cache_groups": 2},
            {"num_prefix_cacheable_groups": 2},
            {"dcp_size": 2},
            {"pcp_size": 2},
            {"pp_size": 2},
        ):
            with self.subTest(overrides=overrides):
                self.assertFalse(evidence_enabled("repair", **{**kw, **overrides}))

    def test_missing_contract_stays_off(self):
        kw = self._ok_kwargs()
        kw["contract_importable"] = False
        self.assertFalse(evidence_enabled("repair", **kw))

    def test_contract_available_matches_environment(self):
        # Under this suite the stand-in contract is installed, so the
        # import guard sees a usable contract.
        self.assertTrue(contract_available())


class _WorkerHarnessMixin:
    """Builds a blockwise KVPoolWorker with the evidence path ACTIVE."""

    def _make_evidence_worker(self, extra_config=None):
        from tests.ut.distributed.ascend_store.test_pool_worker import (
            TestKVPoolWorkerRegisterAndTransfer,
        )

        harness = TestKVPoolWorkerRegisterAndTransfer("test_start_load_kv_sync")
        harness.setUp()
        self.addCleanup(harness.tearDown)
        harness._patch_all()
        cfg = harness._make_config(
            kv_role="kv_consumer",
            extra_config={"backend": "mooncake", "load_async": False, **(extra_config or {})},
        )
        cfg.kv_transfer_config.kv_load_failure_policy = "repair"
        cfg.parallel_config.prefill_context_parallel_size = 1
        cfg.parallel_config.decode_context_parallel_size = 1
        cfg.parallel_config.pipeline_parallel_size = 1
        worker = KVPoolWorker(cfg, use_layerwise=False, kv_cache_config=make_resolved_cache_config(cfg))
        worker.kv_cache_config = None
        # After construction the worker derives evidence facts from the
        # config; fix the layer count to the battery's bound.
        worker._evidence_num_layers = LAYERS
        worker.m_store = MagicMock()
        return worker

    def _req(self, req_id="r1", gen=1, n=N_BLOCKS):
        load_spec = LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=n * BLOCK,
            can_load=True,
            transfer_generation=gen,
        )
        return ReqMeta(
            req_id=req_id,
            token_len_chunk=n * BLOCK,
            block_ids=[100 + i for i in range(n)],
            block_hashes=[f"h{i}" for i in range(n)],
            load_spec=load_spec,
        )

    def _meta(self, reqs):
        meta = AscendConnectorMetadata(set(), set())
        for req in reqs:
            meta.add_request(req)
        return meta


class TestSyncLoadEvidence(_WorkerHarnessMixin, unittest.TestCase):
    """KVPoolWorker.start_load_kv (sync blockwise path) with an active
    evidence path: realistic GET result folding and flat-channel
    suppression."""

    def test_healthy_sync_load_attests_and_suppresses_flat_channel(self):
        # [1] healthy transfer: terminal attestation, no invalid blocks.
        worker = self._make_evidence_worker()
        self.assertIsNotNone(worker.evidence_buffer)
        worker.m_store.get = MagicMock(return_value=[0] * N_BLOCKS)
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req()]))
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())
        self.assertEqual(worker.get_load_failure_reports(), [])
        attestations = worker.get_load_attestations()
        self.assertEqual(len(attestations), 1)
        att = attestations[0]
        self.assertTrue(att.terminal and att.well_formed())
        self.assertEqual(att.loaded_objects, N_BLOCKS * LAYERS)
        self.assertEqual(att.failed_objects, 0)
        # Drained exactly once.
        self.assertEqual(worker.get_load_attestations(), [])

    def test_missing_object_sync_load_single_region_report(self):
        # [2] one object missing: report names the block, all layers.
        worker = self._make_evidence_worker()
        worker.m_store.get = MagicMock(return_value=[0, 0, RC_OBJECT_NOT_FOUND, 0])
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req()]))
        # Flat channel suppressed under the evidence path.
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())
        (report,) = worker.get_load_failure_reports()
        self.assertEqual(report.failed_block_ids, [102])
        self.assertEqual(report.token_span, (32, 48))
        self.assertEqual(report.rollback_boundary, 32)
        (att,) = worker.get_load_attestations()
        self.assertEqual(att.failed_objects, LAYERS)

    def test_whole_batch_exception_fails_all_expected_blocks(self):
        # [3] backend caught an exception: every expected block failed;
        # unrepresentable (multi-block) -> no report; flat channel
        # suppressed.
        worker = self._make_evidence_worker()
        worker.m_store.get = MagicMock(return_value=None)
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req()]))
        self.assertEqual(worker.get_load_failure_reports(), [])
        (att,) = worker.get_load_attestations()
        self.assertEqual(att.failed_objects, N_BLOCKS * LAYERS)
        self.assertEqual(att.loaded_objects, 0)
        self.assertTrue(att.well_formed())
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_whole_batch_exception_with_partial_hash_hits_keeps_the_evidence_fold(self):
        # [3'] main's fine-grained prefix matching (#21167) turns on
        # enable_partial_hash_hits and widens the flat whole-batch predicate;
        # the evidence path folds the failure before that predicate runs.
        worker = self._make_evidence_worker()
        worker.enable_partial_hash_hits = True
        worker.m_store.get = MagicMock(return_value=None)
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req()]))
        self.assertEqual(worker.get_load_failure_reports(), [])
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())
        (att,) = worker.get_load_attestations()
        self.assertEqual(att.failed_objects, N_BLOCKS * LAYERS)
        self.assertTrue(att.well_formed())

    def test_corrupted_object_same_fold_as_missing(self):
        # [2] EIO-class rc on one key: identical single-region fold.
        worker = self._make_evidence_worker()
        worker.m_store.get = MagicMock(return_value=[RC_FILE_READ_FAIL, 0, 0, 0])
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req()]))
        (report,) = worker.get_load_failure_reports()
        self.assertEqual(report.failed_block_ids, [100])
        self.assertEqual(report.rollback_boundary, 0)  # first block failed

    def test_multiple_requests_one_step_drain_both(self):
        # [1/2] multiple GET batches: per-request attempts stay separate.
        worker = self._make_evidence_worker()
        worker.m_store.get = MagicMock(side_effect=[[0] * N_BLOCKS, [0, 0, 0, RC_OBJECT_NOT_FOUND]])
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(
            self._meta(
                [
                    self._req(req_id="r-a", gen=1),
                    self._req(req_id="r-b", gen=2),
                ]
            )
        )
        reports = worker.get_load_failure_reports()
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].request_id, "r-b")
        self.assertEqual(len(worker.get_load_attestations()), 2)

    def test_retry_success_new_generation(self):
        # [5->1] retry with a NEW generation succeeds: the old attempt's
        # evidence is superseded, the new one attests healthy.
        worker = self._make_evidence_worker()
        worker.m_store.get = MagicMock(side_effect=[[0, 0, RC_OBJECT_NOT_FOUND, 0], [0] * N_BLOCKS])
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req(gen=1)]))
        (rep1,) = worker.get_load_failure_reports()
        worker.get_load_attestations()
        self.assertEqual(rep1.transfer_generation, 1)
        worker.start_load_kv(self._meta([self._req(gen=2)]))
        self.assertEqual(worker.get_load_failure_reports(), [])
        (att2,) = worker.get_load_attestations()
        self.assertEqual(att2.transfer_generation, 2)
        self.assertEqual(att2.failed_objects, 0)

    def test_retry_failure_new_generation_fresh_failure(self):
        # [2] a genuine repeated failure carries a NEW generation.
        worker = self._make_evidence_worker()
        worker.m_store.get = MagicMock(side_effect=[[RC_OBJECT_NOT_FOUND, 0, 0, 0], [RC_OBJECT_NOT_FOUND, 0, 0, 0]])
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req(gen=1)]))
        (rep1,) = worker.get_load_failure_reports()
        worker.get_load_attestations()
        worker.start_load_kv(self._meta([self._req(gen=2)]))
        (rep2,) = worker.get_load_failure_reports()
        self.assertEqual(rep2.transfer_generation, 2)
        self.assertEqual(rep1.transfer_generation, 1)

    def test_get_finished_drops_preempted_and_finished_evidence(self):
        # [3] cancellation/preemption cleanup: finished and preempted
        # requests leave no stale buffered evidence.
        worker = self._make_evidence_worker()
        ev = make_attempt(gen=1)
        worker.evidence_buffer.start_attempt(ev)
        # An unrelated finished request and the evidence-holding request
        # both being preempted must drop the buffered evidence.
        meta = AscendConnectorMetadata(preempted_req_ids={"r1", "r2"}, loading_req_ids=set())
        worker.get_finished({"r3"}, meta)
        self.assertIsNone(worker.evidence_buffer.get_attempt("r1"))

    def test_no_identity_attempt_keeps_flat_channel(self):
        # [4] a load spec without a generation (unscheduled attempt)
        # never enters the evidence path: the flat channel owns it.
        worker = self._make_evidence_worker()
        worker.m_store.get = MagicMock(return_value=[RC_OBJECT_NOT_FOUND, 0, 0, 0])
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req(gen=None)]))
        self.assertEqual(worker.get_block_ids_with_load_errors(), {100})
        self.assertEqual(worker.get_load_failure_reports(), [])
        self.assertEqual(worker.get_load_attestations(), [])

    def test_memcache_backend_forced_none_certification(self):
        # [2] the memcache store exposes no service-side write-once
        # binding (client-side exists-filter only; no atomic-visibility
        # contract on the blockwise put), so it can never claim "content"
        # -- underclaim rather than overclaim.
        worker = self._make_evidence_worker(extra_config={"backend": "memcache", "evidence_certification": "content"})
        self.assertIsNotNone(worker.evidence_buffer)
        self.assertEqual(worker._evidence_certification, "none")

    def test_yuanrong_backend_forced_none_certification(self):
        # [2] yuanrong re-PUTs unconditionally: binding not write-once.
        worker = self._make_evidence_worker(extra_config={"backend": "yuanrong", "evidence_certification": "content"})
        self.assertIsNotNone(worker.evidence_buffer)
        self.assertEqual(worker._evidence_certification, "none")

    def test_mooncake_backend_keeps_configured_content(self):
        # [2] the mooncake store carries the documented store-trust
        # assumption (same wrapper and claim as the upstream producer).
        worker = self._make_evidence_worker()  # backend: mooncake
        self.assertEqual(worker._evidence_certification, "content")


class TestAsyncLoadEvidence(unittest.TestCase):
    """KVCacheStoreRecvingThread (async blockwise path) with an active
    evidence path."""

    def _make_thread(self, store):
        db = ChunkedTokenDatabase([KeyMetadata("m", 0, 0, 0)], [BLOCK], None)
        db.set_group_buffers({0: [1000]}, {0: [BLOCK]}, group_num_layers={0: LAYERS})
        buf = EvidenceBuffer()
        worker = MagicMock()
        worker.tp_mismatch = False
        worker._evidence_num_layers = LAYERS
        worker._evidence_certification = "content"
        thread = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=BLOCK,
            tp_rank=0,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
            worker=worker,
            evidence_buffer=buf,
        )
        return thread, buf

    def _req(self, req_id="r1", gen=1):
        load_spec = LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=TOKEN_LEN,
            can_load=True,
            transfer_generation=gen,
        )
        load_spec.token_len = TOKEN_LEN
        return ReqMeta(
            req_id=req_id,
            token_len_chunk=TOKEN_LEN,
            block_ids=[100 + i for i in range(N_BLOCKS)],
            block_hashes=[f"h{i}" for i in range(N_BLOCKS)],
            load_spec=load_spec,
        )

    def test_async_healthy_completion(self):
        # [1] healthy async: terminal attestation after the finished signal.
        store = RcStore([0] * N_BLOCKS)
        thread, buf = self._make_thread(store)
        thread.request_queue.put(self._req())
        thread._handle_request(self._req())
        self.assertIn("r1", thread.get_and_clear_finished_requests())
        (ev,) = buf.drain_terminal()
        att = ev.to_attestation()
        self.assertTrue(att.terminal)
        self.assertEqual(att.loaded_objects, N_BLOCKS * LAYERS)
        self.assertEqual(att.failed_objects, 0)

    def test_async_per_key_failure_report(self):
        # [2] missing object mid-batch: single-region report; the flat
        # channel stays untouched.
        store = RcStore([0, 0, RC_OBJECT_NOT_FOUND, 0])
        thread, buf = self._make_thread(store)
        thread.request_queue.put(self._req())
        thread._handle_request(self._req())
        (ev,) = buf.drain_terminal()
        rep = ev.to_failure_report()
        self.assertEqual(rep.failed_block_ids, [102])
        self.assertEqual(thread._invalid_block_ids, set())

    def test_async_whole_batch_exception_fails_all(self):
        # [3] backend exception: every expected block failed honestly.
        store = RcStore([None])
        thread, buf = self._make_thread(store)
        thread.request_queue.put(self._req())
        thread._handle_request(self._req())
        (ev,) = buf.drain_terminal()
        att = ev.to_attestation()
        self.assertEqual(att.failed_objects, N_BLOCKS * LAYERS)
        self.assertEqual(att.loaded_objects, 0)

    def test_async_exception_handler_drops_identity_matched_evidence(self):
        # [3/4] exception during cleanup: the crashed attempt's entry is
        # dropped; the delivery later fails closed on the missing
        # attestation, never by inheriting a fresh identity.
        store = ExplodingStore()
        thread, buf = self._make_thread(store)
        req = self._req(gen=5)
        try:
            thread.request_queue.put(req)
            # run() takes the item before dispatching it.
            assert thread.request_queue.get() is req
            thread._handle_request(req)
        except RuntimeError:
            thread._handle_request_exception(req)
        self.assertIsNone(buf.get_attempt("r1"))

    def test_async_crash_cannot_destroy_newer_attempt(self):
        # [5] late async completion of an older attempt must not clobber
        # the newer attempt's registered evidence.
        store = ExplodingStore()
        thread, buf = self._make_thread(store)
        old_req = self._req(gen=1)
        newer = make_attempt(gen=2)
        buf.start_attempt(newer)
        try:
            thread.request_queue.put(old_req)
            # run() takes the item before dispatching it.
            assert thread.request_queue.get() is old_req
            thread._handle_request(old_req)
        except RuntimeError:
            thread._handle_request_exception(old_req)
        self.assertEqual(buf.get_attempt("r1").transfer_generation, 2)

    def test_async_stale_evidence_cannot_clobber_newer_delivery(self):
        # [5] a stale terminal attempt registering AFTER a newer in-flight
        # attempt is rejected by the buffer.
        buf = EvidenceBuffer()
        newer = make_attempt(gen=2)
        buf.start_attempt(newer)
        old = fill_attempt(make_attempt(gen=1), failed=(1,))
        old.finalize()
        buf.start_attempt(old)  # late registration of a stale attempt
        current = buf.get_attempt("r1")
        self.assertEqual(current.transfer_generation, 2)
        self.assertFalse(current.terminal)

    def test_async_vacuous_attempt_finalizes_terminal(self):
        # [4] empty expected set: terminal but vacuous (never counts
        # toward the completeness gate).
        store = RcStore([])
        thread, buf = self._make_thread(store)
        load_spec = LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=0,
            can_load=True,
            transfer_generation=1,
        )
        load_spec.token_len = 0
        req = ReqMeta(
            req_id="r-empty",
            token_len_chunk=0,
            block_ids=[],
            block_hashes=[],
            load_spec=load_spec,
        )
        thread.request_queue.put(req)
        thread._handle_request(req)
        (ev,) = buf.drain_terminal()
        att = ev.to_attestation()
        self.assertTrue(att.terminal)
        self.assertEqual(att.expected_objects, 0)
        self.assertFalse(att.certifies_content())

    def test_async_malformed_result_length_fails_all(self):
        # [3] a wrong-shaped backend result makes per-block accounting
        # impossible: every expected block fails honestly.
        store = RcStore([0, 0])  # 2 results for 4 keys
        thread, buf = self._make_thread(store)
        thread.request_queue.put(self._req())
        thread._handle_request(self._req())
        (ev,) = buf.drain_terminal()
        att = ev.to_attestation()
        self.assertEqual(att.failed_objects, N_BLOCKS * LAYERS)
        self.assertEqual(att.loaded_objects, 0)

    def test_async_retry_failure_keeps_generations_separate(self):
        # [2/5] retry -> failure with a fresh generation.
        store = RcStore([0, 0, RC_OBJECT_NOT_FOUND, 0])
        thread, buf = self._make_thread(store)
        thread.request_queue.put(self._req(gen=1))
        thread._handle_request(self._req(gen=1))
        buf.drain_terminal()
        store2 = RcStore([RC_OBJECT_NOT_FOUND, 0, 0, 0])
        thread.m_store = store2
        retry_req = self._req(gen=2)
        thread.request_queue.put(retry_req)
        thread._handle_request(retry_req)
        (ev,) = buf.drain_terminal()
        rep = ev.to_failure_report()
        self.assertEqual(rep.transfer_generation, 2)
        self.assertEqual(rep.failed_block_ids, [100])


class TestFlatChannelCompatibility(_WorkerHarnessMixin, unittest.TestCase):
    """#9701 byte-identity when the evidence path is INACTIVE."""

    def _make_worker(self, policy=None):
        from tests.ut.distributed.ascend_store.test_pool_worker import (
            TestKVPoolWorkerRegisterAndTransfer,
        )

        harness = TestKVPoolWorkerRegisterAndTransfer("test_start_load_kv_sync")
        harness.setUp()
        self.addCleanup(harness.tearDown)
        harness._patch_all()
        cfg = harness._make_config(
            kv_role="kv_consumer",
            extra_config={"backend": "mooncake", "load_async": False},
        )
        if policy is not None:
            cfg.kv_transfer_config.kv_load_failure_policy = policy
        worker = KVPoolWorker(cfg, use_layerwise=False, kv_cache_config=make_resolved_cache_config(cfg))
        worker.kv_cache_config = None
        worker.m_store = MagicMock()
        return worker

    def _req(self, n=N_BLOCKS):
        load_spec = LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=n * BLOCK,
            can_load=True,
        )
        return ReqMeta(
            req_id="r1",
            token_len_chunk=n * BLOCK,
            block_ids=[100 + i for i in range(n)],
            block_hashes=[f"h{i}" for i in range(n)],
            load_spec=load_spec,
        )

    def test_fail_policy_reports_invalid_blocks_exactly_as_before(self):
        # [2] default policy: per-block failures land in the flat channel.
        worker = self._make_worker(policy="fail")
        worker.m_store.get = MagicMock(return_value=[0, 0, RC_OBJECT_NOT_FOUND, 0])
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req()]))
        self.assertEqual(worker.get_block_ids_with_load_errors(), {102})
        self.assertIsNone(worker.evidence_buffer)

    def test_recompute_policy_same_flat_behavior(self):
        worker = self._make_worker(policy="recompute")
        worker.m_store.get = MagicMock(return_value=[0, 0, RC_OBJECT_NOT_FOUND, 0])
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req()]))
        self.assertEqual(worker.get_block_ids_with_load_errors(), {102})
        self.assertIsNone(worker.evidence_buffer)

    def test_mocked_config_default_keeps_flat_channel(self):
        # A config object without the policy attribute (older vLLM) reads
        # as not-repair: flat channel, no evidence buffer.
        worker = self._make_worker()
        worker.m_store.get = MagicMock(return_value=[0, 0, RC_OBJECT_NOT_FOUND, 0])
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        worker.start_load_kv(self._meta([self._req()]))
        self.assertEqual(worker.get_block_ids_with_load_errors(), {102})
        self.assertIsNone(worker.evidence_buffer)

    def test_whole_batch_failure_with_partial_hash_hits_fails_the_request(self):
        # main's #21167 widened the flat whole-batch predicate with
        # enable_partial_hash_hits at the call site this PR replaced by
        # _record_load_get_outcome; the helper carries the same predicate:
        # a backend without per-request failure semantics still fails the
        # whole request instead of only reporting per-block errors.
        worker = self._make_worker(policy="fail")
        worker.enable_partial_hash_hits = True
        worker.m_store.get = MagicMock(return_value=None)
        worker.token_database.set_group_buffers({0: [1000]}, {0: [BLOCK]})
        request = self._req()
        request.can_save = True

        worker.start_load_kv(self._meta([request]))

        self.assertFalse(request.can_save)
        self.assertTrue(worker._save_disabled.is_set())
        self.assertEqual(worker.get_block_ids_with_load_errors(), set(request.block_ids))

    def test_hybrid_layout_with_repair_policy_stays_flat(self):
        # Unsupported layout under repair: evidence off (the worker's gate
        # already ran at construction with the mocked config's facts; pin
        # the pure gate decision here).
        self.assertFalse(
            evidence_enabled(
                "repair",
                contract_importable=True,
                num_prefix_cacheable_groups=1,
                num_kv_cache_groups=2,
                use_hybrid=True,
                use_layerwise=False,
            )
        )


class TestRecordFailedBlocks(unittest.TestCase):
    def test_per_key_codes(self):
        self.assertEqual(record_failed_blocks([1, 2, 3], [0, -704, 0]), {2})
        self.assertEqual(record_failed_blocks([1, 2], [1, 1]), {1, 2})


if __name__ == "__main__":
    unittest.main()
