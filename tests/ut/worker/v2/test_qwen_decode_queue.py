# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Key CPU regressions for decode admission and device-event ordering."""

import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORKER = ROOT / "vllm_ascend/worker/v2"


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


MM = load("_queue_test_mm", WORKER / "mm_decode_queue.py")
PD = load("_queue_test_pd", WORKER / "pd_decode_queue.py")
QUEUE_KEYS = ("enable_qwen_mrv2_early_queue", "enable_qwen_mrv2_mm_decode_early_queue")
MODE = NS(FULL=object())
STREAM = NS(wait_event=Mock())
NAMESPACE: dict[str, Any] = {
    "np": np,
    "torch": NS(npu=NS(current_stream=lambda: STREAM)),
    "SchedulerOutput": object,
    "CUDAGraphMode": MODE,
    "BatchReqState": object,
    "BatchExecutionDescriptor": object,
}
# Execute the production methods unchanged without importing NPU dependencies.
runner_tree = ast.parse((WORKER / "model_runner.py").read_text(encoding="utf-8"))
runner_class = next(
    node for node in runner_tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
)
methods = {
    "_update_seq_lens_cpu",
    "_qwen_mrv2_has_local_uniform_verification",
    "_init_qwen_mrv2_early_queue_flags",
    "_qwen_mrv2_can_queue_decode",
}
for method in runner_class.body:
    if isinstance(method, ast.FunctionDef) and method.name in methods:
        tree = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        exec(compile(tree, str(WORKER / "model_runner.py"), "exec"), NAMESPACE)


def pd_step(**changes: Any) -> Any:
    output = NS(
        kv_connector_metadata=NS(requests={}, requests_to_send={}, reqs_in_batch=set()),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=NS(resumed_req_ids=set()),
        scheduled_new_reqs=[],
        num_scheduled_tokens={"r": 4},
    )
    vars(output).update(changes)
    return output


def receive(state: Any) -> None:
    output = pd_step()
    output.kv_connector_metadata.requests = {"r": object()}
    state.observe_step(output, metadata_known=True)


def completion(**changes: Any) -> Any:
    output = NS(finished_recving={"r"}, failed_recving=set(), invalid_block_ids=set())
    vars(output).update(changes)
    return output


def ready_pd() -> Any:
    state = PD.PDDecodeQueueState()
    receive(state)
    state.observe_completion(completion())
    state.observe_step(pd_step(), metadata_known=True)
    state.record_snapshot({"r": 2}, 10)
    assert state.certify(["r"], [2], [100], [100], 10)
    return state


class PinnedSnapshot:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __getitem__(self, index: int) -> int:
        assert "sync" in self.calls, "pinned snapshot read before its event completed"
        self.calls.append("read")
        return 100


class TestQwenDecodeQueue(unittest.TestCase):
    def test_defaults_and_independent_opt_ins(self):
        modules: dict[str, Any] = {
            name: ModuleType(name)
            for name in (
                "vllm",
                "vllm.logger",
                "vllm.utils",
                "vllm.utils.math_utils",
                "vllm_ascend",
                "vllm_ascend._build_info",
            )
        }
        modules["vllm.logger"].logger = Mock()
        modules["vllm.utils.math_utils"].cdiv = lambda a, b: (a + b - 1) // b
        modules["vllm_ascend"].__path__ = [str(ROOT / "vllm_ascend")]
        modules["vllm_ascend._build_info"].__device_type__ = "A3"
        spec = importlib.util.spec_from_file_location("_queue_test_config", ROOT / "vllm_ascend/ascend_config.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        modules[spec.name] = module
        # Keep the actual complete schema loaded while pydantic resolves forward references.
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
            config_type = module.__dict__["AscendConfig"]
            defaults = dict(sparse_kv_offload_config=NS(enabled=False), multistream_engram_overlap=False)
            for payload, expected in (
                ({}, (False, False)),
                ({QUEUE_KEYS[0]: True}, (True, False)),
                ({QUEUE_KEYS[1]: True}, (False, True)),
                (dict.fromkeys(QUEUE_KEYS, True), (True, True)),
                (dict.fromkeys(QUEUE_KEYS, "false"), (False, False)),
            ):
                with self.subTest(payload=payload):
                    config = config_type(**defaults, **json.loads(json.dumps(payload)))
                    self.assertEqual(tuple(getattr(config, key) for key in QUEUE_KEYS), expected)
            for invalid in (None, ["not a boolean"], "invalid"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    config_type(**defaults, **{QUEUE_KEYS[0]: invalid})

    def test_only_qwen38_targets_can_opt_in_and_mm_requires_text_switch(self):
        supported = ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")
        other = ("Qwen3_5ForConditionalGeneration", "Qwen3NextForCausalLM", "LlamaForCausalLM", "Qwen4ExpMTP", None)
        configure = NAMESPACE["_init_qwen_mrv2_early_queue_flags"]
        for architecture in (*supported, *other):
            for text, mm in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(architecture=architecture, text=text, mm=mm):
                    config = NS(architecture=architecture, model="Qwen3.8-Flash-Next")
                    runner = NS(model_config=config, ascend_config=NS(**dict(zip(QUEUE_KEYS, (text, mm)))))
                    configure(runner)
                    expected = architecture in supported and text
                    self.assertEqual(runner._qwen_mrv2_early_queue_enabled, expected)
                    self.assertEqual(runner._qwen_mrv2_mm_decode_queue_enabled, expected and mm)
                    if not expected:
                        # Disabled and unrelated models return before model-specific imports/state.
                        self.assertFalse(NAMESPACE["_qwen_mrv2_can_queue_decode"](runner, None, None, None))

    def test_device_wait_avoids_pinned_read_and_fallback_synchronizes(self):
        for early in (True, False):
            with self.subTest(early=early):
                calls: list[str] = []

                event = NS(synchronize=lambda calls=calls: calls.append("sync"))
                runner = NS(
                    speculator=object(),
                    sync_spec_pp_cpu_counts=False,
                    _can_skip_spec_cpu_length_sync=lambda: False,
                    num_computed_tokens_event=event,
                    num_computed_tokens_cpu=PinnedSnapshot(calls),
                    req_states=NS(req_id_to_index={"r": 0}, num_computed_tokens_cpu=np.array([104])),
                    input_buffers=NS(seq_lens_cpu=np.zeros(1, dtype=np.int32)),
                    _qwen_mrv2_early_queue_stats={"host_length_wait_skipped": 0, "length_device_wait_enqueued": 0},
                )
                STREAM.wait_event.reset_mock()
                output = NS(num_scheduled_tokens={"r": 4}, scheduled_cached_reqs=NS(req_ids=["r"]))
                NAMESPACE["_update_seq_lens_cpu"](runner, output, ["r"], allow_cpu_upper_bound=early)
                self.assertEqual(calls, [] if early else ["sync", "read"])
                self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [108 if early else 104])
                self.assertEqual(STREAM.wait_event.call_args_list, [unittest.mock.call(event)] if early else [])

    def test_uniform_admission_accepts_padding_but_rejects_invalid_rows(self):
        runner = NS(
            num_speculative_steps=3, req_states=NS(req_id_to_index={"r": 0}, num_computed_tokens_np=np.zeros(1))
        )
        output = NS(num_scheduled_tokens={"r": 4}, scheduled_spec_decode_tokens={"r": [1, 2, 3]})
        batch = NS(
            req_ids=["r"],
            num_scheduled_tokens=np.array([4]),
            num_draft_tokens_np=np.array([3]),
            idx_mapping_np=np.array([0]),
            prefill_len_np=np.array([100]),
            is_prefilling_np=np.array([False]),
            num_tokens=4,
        )
        descriptor = NS(cg_mode=MODE.FULL, uniform_token_count=4, num_tokens=8, num_reqs=2)
        admit = NAMESPACE["_qwen_mrv2_has_local_uniform_verification"]
        self.assertTrue(admit(runner, output, batch, descriptor))
        for field, value in (("idx_mapping_np", np.array([1])), ("num_draft_tokens_np", np.array([2]))):
            with self.subTest(field=field):
                original = getattr(batch, field)
                setattr(batch, field, value)
                self.assertFalse(admit(runner, output, batch, descriptor))
                setattr(batch, field, original)

    def test_mm_requires_a_matching_exact_prompt_snapshot(self):
        media = [
            NS(data=object(), modality="image", identifier="image", mm_position=NS(offset=20, length=30, is_embed=None))
        ]
        key = MM.make_mm_decode_request_key("r", 0, 100, (1,), media)
        state = MM.MMDecodeCertificateState()
        self.assertFalse(state.certify([key], [100]))
        state.record_snapshot([key])
        self.assertFalse(state.certify([key], [99]))
        self.assertFalse(state.is_certified([key]))
        self.assertTrue(state.certify([key], [100]))
        self.assertTrue(state.is_certified([key]))

    def test_mm_layout_integers_are_normalized_before_bounds_checks(self):
        cases: list[tuple[object, object, bool]] = [
            (kind(20), kind(30), True) for kind in (int, np.int32, np.int64, np.uint64)
        ]
        cases += [
            (True, 30, False),
            (20, True, False),
            (np.bool_(True), 30, False),
            (20.0, 30, False),
            (20, 30.0, False),
            (-1, 30, False),
            (20, 0, False),
            (80, 30, False),
            (np.uint64(2**64 - 1), np.uint64(30), False),
        ]
        for offset, length, valid in cases:
            with self.subTest(offset=offset, length=length):
                media = [
                    NS(
                        data=object(),
                        modality="image",
                        identifier="image",
                        mm_position=NS(offset=offset, length=length, is_embed=None),
                    )
                ]
                key = MM.make_mm_decode_request_key("r", 0, 100, (), media)
                self.assertEqual(key is not None, valid)
                if key is not None:
                    self.assertEqual(key.features[1][0][5:7], (20, 30))
                    self.assertTrue(all(type(value) is int for value in key.features[1][0][5:7]))

    def test_mm_identity_layout_and_lifecycle_changes_revoke_proof(self):
        for change in ("slot", "context", "media", "request", "invalidate"):
            with self.subTest(change=change):
                media = [
                    NS(
                        data=object(),
                        modality="image",
                        identifier="image",
                        mm_position=NS(offset=20, length=30, is_embed=None),
                    )
                ]
                key = MM.make_mm_decode_request_key("r", 0, 100, (1,), media)
                state = MM.MMDecodeCertificateState()
                state.record_snapshot([key])
                self.assertTrue(state.certify([key], [100]))
                if change == "media":
                    media[0].mm_position.offset += 1
                if change == "invalidate":
                    state.invalidate()
                changed = MM.make_mm_decode_request_key(
                    "new" if change == "request" else "r",
                    1 if change == "slot" else 0,
                    100,
                    (2,) if change == "context" else (1,),
                    media,
                )
                self.assertFalse(state.is_certified([changed]))

    def test_pd_receipt_alone_cannot_replace_exact_prompt_proof(self):
        state = PD.PDDecodeQueueState()
        receive(state)
        state.record_snapshot({"r": 2}, 10)
        self.assertFalse(state.can_queue(["r"], [2], [100], 10))
        state.observe_step(pd_step(), metadata_known=True)
        self.assertFalse(state.certify(["r"], [2], [100], [100], 10))
        state.observe_completion(completion())
        self.assertFalse(state.can_queue(["r"], [2], [100], 10))
        self.assertFalse(state.certify(["r"], [2], [100], [99], 10))
        self.assertTrue(state.certify(["r"], [2], [100], [100], 10))
        self.assertTrue(state.can_queue(["r"], [2], [100], 10))

    def test_pd_failures_and_changed_binding_use_exact_fallback(self):
        for change in ("failed", "invalid", "generation", "slot"):
            with self.subTest(change=change):
                state = ready_pd()
                if change in ("failed", "invalid", "generation"):
                    receive(state)
                    state.observe_completion(
                        completion(
                            failed_recving={"r"} if change == "failed" else set(),
                            invalid_block_ids={7} if change == "invalid" else set(),
                        )
                    )
                    state.observe_step(pd_step(), metadata_known=True)
                self.assertFalse(state.can_queue(["r"], [3 if change == "slot" else 2], [100], 10))
                if change != "slot":
                    self.assertFalse(state.certify(["r"], [2], [100], [100], 10))
                    state.record_snapshot({"r": 2}, 11)
                    self.assertEqual(state.certify(["r"], [2], [100], [100], 11), change == "generation")

    def test_pd_late_or_overlapping_receipts_do_not_certify_reused_ids(self):
        for retired in (False, True):
            with self.subTest(retired=retired):
                state = PD.PDDecodeQueueState()
                receive(state)
                if retired:
                    state.observe_step(pd_step(finished_req_ids={"r"}), metadata_known=True)
                receive(state)
                state.observe_completion(completion())
                state.observe_step(pd_step(), metadata_known=True)
                state.record_snapshot({"r": 2}, 11)
                self.assertFalse(state.certify(["r"], [2], [100], [100], 11))
                self.assertFalse(state.can_queue(["r"], [2], [100], 11))
                state.observe_completion(completion())
                state.observe_step(pd_step(), metadata_known=True)
                state.record_snapshot({"r": 2}, 12)
                self.assertEqual(state.snapshot()["outstanding_receives"], 0)
                self.assertFalse(state.certify(["r"], [2], [100], [100], 12))

    def connector(self, inner: Any) -> tuple[Any, Any]:
        with patch.dict(
            sys.modules,
            {
                "vllm.v1.worker.gpu.kv_connector": NS(KVConnector=type("KVConnector", (), {})),
                "vllm_ascend.worker.v2.pd_decode_queue": PD,
            },
        ):
            module = load("_queue_test_connector", WORKER / "kv_connector.py")
        metadata_type = type("Metadata", (), {})
        transfer = NS(get_finished=Mock(side_effect=AssertionError("unexpected polling")))
        wrapper = module.PDDecodeQueueKVConnector(inner, transfer, metadata_type)
        return wrapper, transfer

    def test_connector_observes_existing_callbacks_once_without_polling(self):
        result = completion()
        no_forward_result = NS(kv_connector_output=result)
        inner = NS(post_forward=Mock(return_value=result), no_forward=Mock(return_value=no_forward_result))
        wrapper, transfer = self.connector(inner)
        wrapper.decode_queue_state = Mock()
        self.assertIs(wrapper.post_forward(set()), result)
        wrapper.decode_queue_state.observe_completion.assert_called_once_with(result)
        inner.post_forward.assert_called_once_with(set())
        wrapper.decode_queue_state.reset_mock()
        output = pd_step()
        output.kv_connector_metadata = wrapper._metadata_type()
        vars(output.kv_connector_metadata).update(requests={}, requests_to_send={}, reqs_in_batch=set())
        self.assertIs(wrapper.no_forward(output), no_forward_result)
        inner.no_forward.assert_called_once_with(output)
        wrapper.decode_queue_state.observe_step.assert_called_once_with(output, metadata_known=True)
        wrapper.decode_queue_state.observe_completion.assert_called_once_with(result)
        transfer.get_finished.assert_not_called()

    def test_connector_error_invalidates_and_preserves_original_exception(self):
        error = RuntimeError("receive failed")
        inner = NS(post_forward=Mock(side_effect=error))
        wrapper, transfer = self.connector(inner)
        wrapper.decode_queue_state = Mock()
        with self.assertRaises(RuntimeError) as raised:
            wrapper.post_forward(set())
        self.assertIs(raised.exception, error)
        wrapper.decode_queue_state.invalidate.assert_called_once_with()
        wrapper.decode_queue_state.observe_completion.assert_not_called()
        transfer.get_finished.assert_not_called()

    def test_connector_unknown_metadata_revokes_proof_without_interrupting_forward(self):
        def read_metadata(instance: Any, name: str) -> Any:
            if name == "requests":
                raise RuntimeError("metadata schema changed")
            return object.__getattribute__(instance, name)

        for scenario in ("known", "slots", "extra", "missing", "wrong_type", "unreadable"):
            with self.subTest(scenario=scenario):
                inner = NS(pre_forward=Mock())
                wrapper, transfer = self.connector(inner)
                wrapper.decode_queue_state = ready_pd()
                if scenario == "slots":
                    wrapper._metadata_type = type(
                        "Metadata", (), {"__slots__": ("requests", "requests_to_send", "reqs_in_batch")}
                    )
                if scenario == "unreadable":
                    wrapper._metadata_type = type("Metadata", (), {"__getattribute__": read_metadata})
                metadata = wrapper._metadata_type()
                metadata.requests, metadata.requests_to_send, metadata.reqs_in_batch = {}, {}, set()
                if scenario == "extra":
                    metadata.new_transfer_action = {"r"}
                if scenario == "missing":
                    del metadata.reqs_in_batch
                if scenario == "wrong_type":
                    metadata.requests = []
                output = pd_step(kv_connector_metadata=metadata)
                wrapper.pre_forward(output)
                self.assertEqual(wrapper.decode_queue_state.can_queue(["r"], [2], [100], 10), scenario == "known")
                inner.pre_forward.assert_called_once_with(output)
                transfer.get_finished.assert_not_called()


if __name__ == "__main__":
    unittest.main()
