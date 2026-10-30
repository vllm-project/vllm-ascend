# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU lifecycle tests; no transfer queues, NPU runtime, or model are needed."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
HELPER_PATH = ROOT / "vllm_ascend/worker/v2/pd_decode_queue.py"
SPEC = importlib.util.spec_from_file_location("_test_pd_decode_queue_helper", HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {SPEC.name: HELPER}):
    SPEC.loader.exec_module(HELPER)


def step(*, requests=None, sends=None, in_batch=None, scheduled=None, **changes):
    output = NS(
        kv_connector_metadata=NS(
            requests=requests or {}, requests_to_send=sends or {}, reqs_in_batch=in_batch or set()
        ),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=NS(resumed_req_ids=set()),
        scheduled_new_reqs=[],
        num_scheduled_tokens={"ready": 4} if scheduled is None else scheduled,
    )
    vars(output).update(changes)
    return output


def completion(*, finished=(), failed=(), invalid=()):
    return NS(finished_recving=set(finished), failed_recving=set(failed), invalid_block_ids=set(invalid))


class TestPDReadiness(unittest.TestCase):
    def ready_state(self):
        state = HELPER.PDDecodeQueueState()
        state.observe_step(step(requests={"ready": object()}, scheduled={}), metadata_known=True)
        state.observe_completion(completion(finished={"ready"}))
        state.observe_step(step(), metadata_known=True)
        state.record_snapshot({"ready": 2}, 10)
        self.assertTrue(state.certify(["ready"], [2], [100], [100], 10))
        return state

    def test_receipt_requires_exact_prompt_snapshot_before_queue(self):
        state = HELPER.PDDecodeQueueState()
        state.observe_step(step(requests={"ready": object()}), metadata_known=True)
        state.record_snapshot({"ready": 2}, 10)
        self.assertFalse(state.can_queue(["ready"], [2], [100], 10))
        state.observe_completion(completion(finished={"ready"}))
        state.observe_step(step(), metadata_known=True)
        self.assertFalse(state.can_queue(["ready"], [2], [100], 10))
        self.assertFalse(state.certify(["ready"], [2], [100], [99], 10))
        self.assertTrue(state.certify(["ready"], [2], [100], [100], 10))
        self.assertTrue(state.can_queue(["ready"], [2], [100], 10))

    def test_missing_completion_cannot_be_inferred_from_empty_metadata(self):
        state = HELPER.PDDecodeQueueState()
        state.observe_step(step(requests={"ready": object()}), metadata_known=True)
        for generation in range(1, 5):
            state.observe_step(step(), metadata_known=True)
            state.record_snapshot({"ready": 2}, generation)
            self.assertFalse(state.certify(["ready"], [2], [100], [200], generation))
            self.assertFalse(state.can_queue(["ready"], [2], [100], generation))

    def test_local_prefill_or_prefix_hit_without_remote_transfer_can_certify(self):
        for exact in (100, 140):
            with self.subTest(exact=exact):
                state = HELPER.PDDecodeQueueState()
                state.observe_step(step(), metadata_known=True)
                state.record_snapshot({"ready": 2}, 10)
                self.assertTrue(state.certify(["ready"], [2], [100], [exact], 10))
                self.assertTrue(state.can_queue(["ready"], [2], [100], 10))
                self.assertEqual(state.snapshot()["outstanding_receives"], 0)

    def test_other_request_can_receive_without_blocking_ready_rows(self):
        state = self.ready_state()
        state.observe_step(step(requests={"other": object()}, in_batch={"other"}), metadata_known=True)
        self.assertTrue(state.can_queue(["ready"], [2], [100], 10))
        self.assertEqual(state.snapshot()["pending_requests"], 1)

    def test_failed_or_invalid_receives_revoke_proof(self):
        for output in (completion(finished={"ready"}, failed={"ready"}), completion(invalid={7})):
            with self.subTest(output=output):
                state = self.ready_state()
                if output.failed_recving:
                    state.observe_step(step(requests={"ready": object()}), metadata_known=True)
                state.observe_completion(output)
                state.observe_step(step(), metadata_known=True)
                state.record_snapshot({"ready": 2}, 11)
                self.assertFalse(state.certify(["ready"], [2], [100], [100], 11))
                self.assertFalse(state.can_queue(["ready"], [2], [100], 11))

    def test_invalid_blocks_take_precedence_over_same_batch_completion(self):
        state = HELPER.PDDecodeQueueState()
        state.observe_step(step(requests={"ready": object()}), metadata_known=True)
        state.observe_completion(completion(finished={"ready"}, invalid={7}))
        state.observe_step(step(), metadata_known=True)
        state.record_snapshot({"ready": 2}, 11)
        self.assertFalse(state.certify(["ready"], [2], [100], [100], 11))

    def test_changed_slot_prompt_or_snapshot_generation_requires_exact_path(self):
        for slots, prompts, generation in (([3], [100], 10), ([2], [101], 10), ([2], [100], 11)):
            with self.subTest(slots=slots, prompts=prompts, generation=generation):
                state = self.ready_state()
                self.assertFalse(state.can_queue(["ready"], slots, prompts, generation))

    def test_rebound_slot_requires_a_new_snapshot_and_exact_certificate(self):
        state = self.ready_state()
        self.assertFalse(state.certify(["ready"], [3], [100], [100], 10))
        state.record_snapshot({"ready": 3}, 11)
        self.assertFalse(state.can_queue(["ready"], [3], [100], 11))
        self.assertTrue(state.certify(["ready"], [3], [100], [100], 11))
        self.assertTrue(state.can_queue(["ready"], [3], [100], 11))

    def test_new_transfer_generation_invalidates_previous_snapshot(self):
        state = self.ready_state()
        state.observe_step(step(requests={"ready": object()}), metadata_known=True)
        state.observe_completion(completion(finished={"ready"}))
        state.observe_step(step(), metadata_known=True)
        self.assertFalse(state.can_queue(["ready"], [2], [100], 10))
        self.assertFalse(state.certify(["ready"], [2], [100], [100], 10))
        state.record_snapshot({"ready": 2}, 11)
        self.assertTrue(state.certify(["ready"], [2], [100], [100], 11))

    def test_overlapping_generations_do_not_accept_unattributable_old_receipt(self):
        state = HELPER.PDDecodeQueueState()
        state.observe_step(step(requests={"ready": object()}), metadata_known=True)
        state.observe_step(step(requests={"ready": object()}), metadata_known=True)
        state.observe_completion(completion(finished={"ready"}))
        state.observe_step(step(), metadata_known=True)
        state.record_snapshot({"ready": 2}, 11)
        self.assertFalse(state.certify(["ready"], [2], [100], [100], 11))

    def test_multiple_overlaps_drain_once_per_union_and_keep_lifetime_ambiguous(self):
        for statuses in ((False, False, True), (True, False, False), (False, True, False)):
            with self.subTest(statuses=statuses):
                state = HELPER.PDDecodeQueueState()
                for _ in range(3):
                    state.observe_step(step(requests={"ready": object()}), metadata_known=True)
                for index, failed in enumerate(statuses):
                    # Failed receives are also in finished_recving; this must
                    # consume one notification, not two generation counts.
                    state.observe_completion(completion(finished={"ready"}, failed={"ready"} if failed else ()))
                    state.observe_step(step(), metadata_known=True)
                    state.record_snapshot({"ready": 2}, 11 + index)
                    self.assertFalse(state.certify(["ready"], [2], [100], [100], 11 + index))
                    self.assertEqual(state.snapshot()["outstanding_receives"], 2 - index)
                state.observe_completion(completion(finished={"ready"}))
                self.assertEqual(state.snapshot()["outstanding_receives"], 0)
                self.assertFalse(state.can_queue(["ready"], [2], [100], 13))
                state.observe_step(step(finished_req_ids={"ready"}, scheduled={}), metadata_known=True)
                state.observe_step(step(requests={"ready": object()}, scheduled={}), metadata_known=True)
                state.observe_completion(completion(finished={"ready"}))
                state.observe_step(step(), metadata_known=True)
                state.record_snapshot({"ready": 2}, 20)
                self.assertTrue(state.certify(["ready"], [2], [100], [100], 20))

    def test_cancelled_ids_do_not_accumulate_after_normal_late_receipts(self):
        state = HELPER.PDDecodeQueueState()
        cancelled_ids = {f"cancelled-{index}" for index in range(2048)}
        for req_id in cancelled_ids:
            state.observe_step(step(requests={req_id: object()}, scheduled={}), metadata_known=True)
            state.observe_step(step(finished_req_ids={req_id}, scheduled={}), metadata_known=True)
        self.assertEqual(state.snapshot()["requests"], 0)
        self.assertEqual(state.snapshot()["outstanding_receive_ids"], len(cancelled_ids))
        state.observe_completion(completion(finished=cancelled_ids))
        self.assertEqual(state.snapshot()["outstanding_receive_ids"], 0)
        self.assertEqual(state.snapshot()["outstanding_receives"], 0)
        # Duplicate/unknown notifications cannot underflow or recreate entries.
        state.observe_completion(completion(finished=cancelled_ids | {"unknown"}))
        self.assertEqual(state.snapshot()["requests"], 0)
        self.assertEqual(state.snapshot()["outstanding_receive_ids"], 0)

    def test_late_receipt_cannot_certify_reused_request_id_or_slot(self):
        state = HELPER.PDDecodeQueueState()
        state.observe_step(step(requests={"ready": object()}), metadata_known=True)
        state.observe_step(step(finished_req_ids={"ready"}, scheduled={}), metadata_known=True)
        state.observe_step(step(requests={"ready": object()}), metadata_known=True)
        state.observe_completion(completion(finished={"ready"}))
        state.observe_step(step(), metadata_known=True)
        state.record_snapshot({"ready": 2}, 11)
        self.assertFalse(state.certify(["ready"], [2], [100], [100], 11))

    def test_finished_preempted_and_resumed_requests_lose_certificate(self):
        for changes in (
            {"finished_req_ids": {"ready"}},
            {"preempted_req_ids": {"ready"}},
            {"scheduled_cached_reqs": NS(resumed_req_ids={"ready"})},
            {"scheduled_new_reqs": [NS(req_id="ready")]},
        ):
            with self.subTest(changes=changes):
                state = self.ready_state()
                state.observe_step(step(**changes), metadata_known=True)
                self.assertFalse(state.can_queue(["ready"], [2], [100], 10))

    def test_receipt_before_new_request_admission_is_preserved(self):
        state = HELPER.PDDecodeQueueState()
        state.observe_step(step(requests={"ready": object()}, scheduled={}), metadata_known=True)
        state.observe_completion(completion(finished={"ready"}))
        state.observe_step(step(scheduled_new_reqs=[NS(req_id="ready")]), metadata_known=True)
        state.record_snapshot({"ready": 2}, 10)
        state.observe_step(step(), metadata_known=True)
        self.assertTrue(state.certify(["ready"], [2], [100], [100], 10))

    def test_sends_and_batch_transfer_actions_require_baseline_for_affected_rows(self):
        for changes in ({"sends": {"ready": 0.0}}, {"in_batch": {"ready"}}):
            with self.subTest(changes=changes):
                state = self.ready_state()
                state.observe_step(step(**changes), metadata_known=True)
                self.assertFalse(state.can_queue(["ready"], [2], [100], 10))

    def test_unknown_metadata_or_capture_reset_revokes_old_state(self):
        state = self.ready_state()
        state.observe_step(step(), metadata_known=False)
        self.assertFalse(state.can_queue(["ready"], [2], [100], 10))
        state.observe_step(step(), metadata_known=True)
        state.record_snapshot({"ready": 2}, 11)
        self.assertFalse(state.certify(["ready"], [2], [100], [100], 11))
        state = self.ready_state()
        state.invalidate()
        self.assertFalse(state.can_queue(["ready"], [2], [100], 10))


class TestPDConnectorDecorator(unittest.TestCase):
    def setUp(self):
        class KVConnector:
            pass

        spec = importlib.util.spec_from_file_location(
            "_test_pd_connector_decorator", HELPER_PATH.with_name("kv_connector.py")
        )
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "vllm.v1.worker.gpu.kv_connector": NS(KVConnector=KVConnector),
                "vllm_ascend.worker.v2.pd_decode_queue": HELPER,
            },
        ):
            spec.loader.exec_module(self.module)

        class Metadata:
            def __init__(self):
                self.requests = {}
                self.requests_to_send = {}
                self.reqs_in_batch = set()

        self.Metadata = Metadata

    def wrapper(self):
        original = completion()
        inner = NS(
            pre_forward=Mock(),
            finish_forward=Mock(),
            post_forward=Mock(return_value=original),
            finalize_step=Mock(),
            no_forward=Mock(return_value=NS(kv_connector_output=original)),
            set_disabled=Mock(),
            reset_capture_state=Mock(),
            original_attribute=object(),
        )
        wrapped = self.module.PDDecodeQueueKVConnector(inner, object(), self.Metadata)
        return wrapped, inner

    def test_callbacks_outputs_and_original_attributes_are_preserved(self):
        wrapped, inner = self.wrapper()
        output = step()
        output.kv_connector_metadata = self.Metadata()
        batch, metadata = object(), object()
        wrapped.pre_forward(output, batch, attn_metadata=metadata)
        wrapped.finish_forward()
        result = wrapped.post_forward({"finished"}, defer_finalize=True)
        self.assertIs(result, inner.post_forward.return_value)
        wrapped.finalize_step()
        self.assertIs(wrapped.no_forward(step()), inner.no_forward.return_value)
        self.assertIs(wrapped.original_attribute, inner.original_attribute)
        inner.pre_forward.assert_called_once_with(output, batch, attn_metadata=metadata)
        inner.finish_forward.assert_called_once_with()
        inner.post_forward.assert_called_once_with({"finished"}, defer_finalize=True)
        inner.finalize_step.assert_called_once_with()
        self.assertEqual(wrapped.decode_queue_state.stats["completion_observations"], 2)

    def test_no_forward_observes_receive_and_receipt_once_without_extra_poll(self):
        wrapped, inner = self.wrapper()
        output = step(scheduled={})
        output.kv_connector_metadata = self.Metadata()
        output.kv_connector_metadata.requests["ready"] = object()
        inner.no_forward.return_value.kv_connector_output = completion(finished={"ready"})
        result = wrapped.no_forward(output)
        self.assertIs(result, inner.no_forward.return_value)
        inner.no_forward.assert_called_once_with(output)
        self.assertEqual(wrapped.decode_queue_state.stats["receive_generations"], 1)
        self.assertEqual(wrapped.decode_queue_state.stats["completed_receives"], 1)

    def test_disabled_capture_does_not_invent_completion_proof(self):
        wrapped, inner = self.wrapper()
        wrapped.set_disabled(True)
        wrapped.post_forward(set())
        self.assertEqual(wrapped.decode_queue_state.stats["completion_observations"], 0)
        wrapped.set_disabled(False)
        wrapped.reset_capture_state()
        inner.set_disabled.assert_any_call(True)
        inner.set_disabled.assert_any_call(False)
        inner.reset_capture_state.assert_called_once_with()

    def test_transfer_error_remains_original_error_and_revokes_proof(self):
        for callback in ("pre_forward", "post_forward", "no_forward", "finish_forward", "finalize_step"):
            with self.subTest(callback=callback):
                wrapped, inner = self.wrapper()
                error = RuntimeError("transfer failed")
                getattr(inner, callback).side_effect = error
                with self.assertRaises(RuntimeError) as caught:
                    args = () if callback in ("finish_forward", "finalize_step") else (step(),)
                    getattr(wrapped, callback)(*args)
                self.assertIs(caught.exception, error)
                self.assertGreater(wrapped.decode_queue_state.stats["invalidations"], 0)

    def test_mamba_copy_wrapper_preserves_deferred_finalize(self):
        inner = NS(finalize_step=Mock())
        wrapped = self.module.MambaCopyKVConnector(inner, Mock())
        wrapped.finalize_step()
        inner.finalize_step.assert_called_once_with()

    def test_factory_leaves_unknown_connector_without_importing_mooncake(self):
        inner, group = object(), object()
        self.assertIs(self.module.wrap_pd_decode_queue_connector(inner, group), inner)

    def test_factory_accepts_registered_pull_consumer_alias_and_mamba_wrapper(self):
        class ActiveKVConnector:
            pass

        class MooncakePullConnector:
            pass

        MooncakePullConnector.__module__ = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector"
        group = MooncakePullConnector()
        group.kv_transfer_config = NS(is_kv_consumer=True)
        inner = ActiveKVConnector()
        inner.kv_connector = group
        with patch.dict(
            sys.modules,
            {
                "vllm.v1.worker.gpu.kv_connector": NS(ActiveKVConnector=ActiveKVConnector),
                MooncakePullConnector.__module__: NS(MooncakePullConnector=MooncakePullConnector),
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata": NS(
                    MooncakeConnectorMetadata=self.Metadata
                ),
            },
        ):
            for connector in (inner, self.module.MambaCopyKVConnector(inner, Mock())):
                wrapped = self.module.wrap_pd_decode_queue_connector(connector, group)
                self.assertIsInstance(wrapped, self.module.PDDecodeQueueKVConnector)
                self.assertIs(wrapped._connector, connector)
            group.kv_transfer_config.is_kv_consumer = False
            self.assertIs(self.module.wrap_pd_decode_queue_connector(inner, group), inner)


if __name__ == "__main__":
    unittest.main(verbosity=2)
