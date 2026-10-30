# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Key CPU regressions for model capabilities and shared decode submission."""

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
MODEL = ROOT / "vllm_ascend/models/qwen4_exp"
QUEUE_KEYS = ("enable_decode_early_queue", "enable_mm_decode_early_queue")
FULL = object()


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


MM = load("_decode_test_mm", WORKER / "mm_decode_queue.py")
PD = load("_decode_test_pd", WORKER / "pd_decode_queue.py")
with patch.dict(
    sys.modules, {"vllm_ascend.worker.v2.mm_decode_queue": MM, "vllm_ascend.worker.v2.pd_decode_queue": PD}
):
    QUEUE = load("_decode_test_queue", WORKER / "decode_queue.py")
    QWEN = load("_decode_test_qwen", MODEL / "decode_queue.py")


def pd_step(**changes: Any) -> Any:
    output = NS(
        kv_connector_metadata=NS(requests={}, requests_to_send={}, reqs_in_batch=set()),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=NS(resumed_req_ids=set(), req_ids=["r"]),
        scheduled_new_reqs=[],
        num_scheduled_tokens={"r": 4},
        scheduled_spec_decode_tokens={"r": [1, 2, 3]},
        scheduled_encoder_inputs={},
        free_encoder_mm_hashes=[],
        ec_connector_metadata=None,
        ec_manager_metadata=None,
        kv_cache_block_copies=[],
        has_structured_output_requests=False,
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


def fixture() -> tuple[Any, Any, Any, Any]:
    capability = NS(supports=Mock(return_value=True), mm_context=lambda _: ("media-consumers",))
    runner = NS(
        model_state=NS(get_decode_queue_capability=lambda _: capability),
        num_speculative_steps=3,
        max_model_len=1000,
        req_states=NS(
            req_id_to_index={"r": 0},
            num_computed_tokens_np=np.array([104]),
            num_computed_tokens_cpu=np.array([104]),
            prefill_len=NS(np=np.array([100])),
        ),
        num_computed_tokens_cpu=NS(numpy=lambda: np.array([104])),
        speculator=NS(draft_model_config=NS(max_model_len=1000)),
        speculative_config=NS(method="mtp", uses_dynamic_speculative_decoding=lambda: False),
        parallel_config=NS(decode_context_parallel_size=1, prefill_context_parallel_size=1, enable_dbo=False),
        encoder_cache=None,
        supports_mm_inputs=False,
        model_config=NS(enable_prompt_embeds=False, is_mm_prefix_lm=False),
        lora_config=None,
        aux_output_connector=None,
        adaptive_verification=None,
        sparse_kv_offload_manager=None,
        sync_spec_pp_cpu_counts=False,
        _can_skip_spec_cpu_length_sync=lambda: False,
    )
    batch = NS(
        req_ids=["r"],
        num_scheduled_tokens=np.array([4]),
        num_draft_tokens_np=np.array([3]),
        idx_mapping_np=np.array([0]),
        prefill_len_np=np.array([100]),
        is_prefilling_np=np.array([False]),
        num_tokens=4,
        has_prefill=False,
        prefill_runs_as_decode_np=None,
    )
    desc = NS(cg_mode=FULL, uniform_token_count=4, num_tokens=8, num_reqs=2)
    return runner, pd_step(), batch, desc


def media() -> list[Any]:
    return [
        NS(data=object(), modality="image", identifier="image", mm_position=NS(offset=20, length=30, is_embed=None))
    ]


def qwen_state_module() -> Any:
    with patch.dict(
        sys.modules,
        {
            "vllm.models.qwen4_exp.amd.model_state": NS(Qwen4ExpModelState=type("QwenState", (), {})),
            "vllm_ascend.worker.v2.model_states.mamba_hybrid": NS(
                AscendMambaHybridModelState=type("HybridState", (), {})
            ),
        },
    ):
        return load("_decode_test_qwen_state", MODEL / "model_state.py")


class PinnedSnapshot:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __getitem__(self, index: int) -> int:
        assert "sync" in self.calls, "pinned snapshot read before its event completed"
        self.calls.append("read")
        return 100


class TestDecodeQueue(unittest.TestCase):
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

    def test_capability_admits_valid_decode_independently_of_model_name(self):
        runner, output, batch, desc = fixture()
        controller, stream = QUEUE.DecodeQueueController(True, full_graph_mode=FULL), object()
        controller.record_snapshot(runner, stream)
        self.assertTrue(controller.can_queue(runner, output, batch, desc, stream, connector_ready=True))
        runner.model_state.get_decode_queue_capability(runner).supports.assert_called_once_with(runner)
        runner.model_state = NS()
        self.assertFalse(controller.can_queue(runner, output, batch, desc, stream, connector_ready=True))
        self.assertEqual(controller.last_guard["failed"], ["model_capability"])

    def test_disabled_and_unknown_capabilities_do_not_read_optional_state(self):
        class IncompleteRunner:
            def __getattr__(self, name: str) -> Any:
                raise AssertionError(f"disabled queue accessed {name}")

        controller = QUEUE.DecodeQueueController()
        runner = IncompleteRunner()
        self.assertFalse(controller.model_enabled(runner))
        self.assertFalse(controller.can_queue(runner, None, None, None, None))
        controller.record_snapshot(runner, None)
        controller.certify_exact(runner, None, None, None)
        controller.before_requests(None)
        self.assertEqual(controller.snapshot_generation, 0)
        self.assertFalse(QUEUE.DecodeQueueController(True).can_queue(NS(), None, None, None, None))
        enabled = QUEUE.DecodeQueueController(True)
        faulty = NS(model_state=NS(get_decode_queue_capability=Mock(side_effect=RuntimeError("unsupported"))))
        self.assertFalse(enabled.model_enabled(faulty))
        self.assertFalse(QUEUE.DecodeQueueController(False, True).multimodal)

    def test_uniform_padding_and_invalid_rows(self):
        runner, output, batch, desc = fixture()
        controller = QUEUE.DecodeQueueController(True, full_graph_mode=FULL)
        self.assertTrue(controller.has_local_uniform_verification(runner, output, batch, desc))
        for field, value in (
            ("idx_mapping_np", np.array([1])),
            ("idx_mapping_np", np.array([0.0])),
            ("num_draft_tokens_np", np.array([2])),
            ("num_scheduled_tokens", np.array([4, 4])),
        ):
            with self.subTest(field=field, value=value):
                original = getattr(batch, field)
                setattr(batch, field, value)
                self.assertFalse(controller.has_local_uniform_verification(runner, output, batch, desc))
                setattr(batch, field, original)

    def test_runtime_guards_and_capability_errors_keep_exact_fallback(self):
        for change in ("arrival", "stream", "lora", "overflow", "capability"):
            with self.subTest(change=change):
                runner, output, batch, desc = fixture()
                controller, stream = QUEUE.DecodeQueueController(True, full_graph_mode=FULL), object()
                controller.record_snapshot(runner, stream)
                if change == "arrival":
                    output.scheduled_new_reqs = [NS(req_id="new")]
                if change == "stream":
                    stream = object()
                if change == "lora":
                    runner.lora_config = object()
                if change == "overflow":
                    runner.max_model_len = 106
                if change == "capability":
                    runner.model_state.get_decode_queue_capability(runner).supports.side_effect = RuntimeError(
                        "unknown"
                    )
                self.assertFalse(controller.can_queue(runner, output, batch, desc, stream, connector_ready=True))

    def test_device_wait_never_reads_pinned_snapshot_and_fallback_synchronizes(self):
        for early in (True, False):
            with self.subTest(early=early):
                calls: list[str] = []
                runner, output, _, _ = fixture()
                runner.num_computed_tokens_cpu = PinnedSnapshot(calls)
                runner.num_computed_tokens_event = NS(synchronize=lambda calls=calls: calls.append("sync"))
                runner.input_buffers = NS(seq_lens_cpu=np.zeros(1, dtype=np.int32))
                controller, stream = QUEUE.DecodeQueueController(True), NS(wait_event=Mock())
                QUEUE.update_seq_lens_cpu(
                    runner, output, ["r"], allow_cpu_upper_bound=early, current_stream=stream, controller=controller
                )
                self.assertEqual(calls, [] if early else ["sync", "read"])
                self.assertEqual(runner.input_buffers.seq_lens_cpu.tolist(), [108 if early else 104])
                self.assertEqual(stream.wait_event.call_count, int(early))
                self.assertEqual(controller.stats["host_length_wait_skipped"], int(early))

    def test_multimodal_controller_requires_exact_snapshot_and_revokes_lifecycle(self):
        runner, output, batch, desc = fixture()
        runner.encoder_cache, runner.supports_mm_inputs = NS(mm_features={"r": media()}), True
        controller, stream = QUEUE.DecodeQueueController(True, True, FULL), object()
        controller.record_snapshot(runner, stream)
        self.assertFalse(controller.can_queue(runner, output, batch, desc, stream, connector_ready=True))
        controller.certify_exact(runner, output, batch, stream)
        self.assertTrue(controller.can_queue(runner, output, batch, desc, stream, connector_ready=True))
        output.free_encoder_mm_hashes = ["image"]
        controller.before_requests(output)
        self.assertFalse(controller.can_queue(runner, output, batch, desc, stream, connector_ready=True))
        self.assertEqual(controller.multimodal_stats["invalidations"], 1)

    def test_pd_controller_requires_completed_receive_and_exact_snapshot(self):
        runner, output, batch, desc = fixture()
        controller, stream, state = (
            QUEUE.DecodeQueueController(True, full_graph_mode=FULL),
            object(),
            PD.PDDecodeQueueState(),
        )
        receive(state)
        state.observe_step(output, metadata_known=True)
        controller.record_snapshot(runner, stream, state)
        controller.certify_exact(runner, output, batch, stream, state)
        self.assertFalse(
            controller.can_queue(runner, output, batch, desc, stream, pd_state=state, pd_transfer_group_matches=True)
        )
        state.observe_completion(completion())
        controller.certify_exact(runner, output, batch, stream, state)
        self.assertTrue(
            controller.can_queue(runner, output, batch, desc, stream, pd_state=state, pd_transfer_group_matches=True)
        )
        self.assertFalse(
            controller.can_queue(runner, output, batch, desc, stream, pd_state=state, pd_transfer_group_matches=False)
        )

    def test_changed_model_owner_or_missing_capability_revokes_pd_certificate(self):
        for change in ("owner", "capability"):
            with self.subTest(change=change):
                runner, output, batch, desc = fixture()
                controller, stream = QUEUE.DecodeQueueController(True, full_graph_mode=FULL), object()
                state = PD.PDDecodeQueueState()
                receive(state)
                state.observe_completion(completion())
                state.observe_step(output, metadata_known=True)
                controller.record_snapshot(runner, stream, state)
                controller.certify_exact(runner, output, batch, stream, state)
                self.assertTrue(
                    controller.can_queue(
                        runner, output, batch, desc, stream, pd_state=state, pd_transfer_group_matches=True
                    )
                )
                owner, factory = runner.model_state, runner.model_state.get_decode_queue_capability
                if change == "owner":
                    runner.model_state = NS(get_decode_queue_capability=factory)
                else:
                    owner.get_decode_queue_capability = lambda _: None
                self.assertEqual(controller.model_enabled(runner), change == "owner")
                self.assertEqual(state.snapshot()["certified_requests"], 0)
                # Restoring the old owner and reusing generation 1 must not
                # resurrect its stale transfer/prompt certificate.
                runner.model_state, owner.get_decode_queue_capability = owner, factory
                controller.record_snapshot(runner, stream, state)
                self.assertFalse(
                    controller.can_queue(
                        runner, output, batch, desc, stream, pd_state=state, pd_transfer_group_matches=True
                    )
                )

    def test_qwen_state_offers_capability_only_to_supported_architectures(self):
        module = qwen_state_module()
        state = module.AscendQwen4ExpModelState()
        with patch.dict(sys.modules, {"vllm_ascend.models.qwen4_exp.decode_queue": QWEN}):
            for architecture in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration", "LlamaForCausalLM", None):
                with self.subTest(architecture=architecture):
                    runner = NS(model_state=state, model_config=NS(architecture=architecture))
                    capability = state.get_decode_queue_capability(runner)
                    self.assertEqual(
                        capability is not None, architecture is not None and architecture.startswith("Qwen4Exp")
                    )
            self.assertIsNone(state.get_decode_queue_capability(NS()))
            subclass = type("CustomQwenState", (module.AscendQwen4ExpModelState,), {})()
            self.assertIsNone(subclass.get_decode_queue_capability(NS(model_state=subclass)))
        self.assertFalse(QWEN.QwenDecodeQueueCapability().supports(NS()))
        self.assertIsNone(QWEN.QwenDecodeQueueCapability().mm_context(NS()))

    def test_qwen_capability_checks_target_and_draft_inventory_each_time(self):
        state_module = qwen_state_module()
        impl, builder, state_builder = type("Impl", (), {}), type("Builder", (), {}), type("StateBuilder", (), {})
        backend = type("Backend", (), {"get_impl_cls": staticmethod(lambda: impl)})
        speculator_type = type("Speculator", (), {})
        names = (
            "AscendGDNAttentionBackend",
            "AscendGDNAttentionMetadataBuilder",
            "AscendGDNFusedAttentionBackend",
            "AscendGDNHostMetadataBackend",
            "AscendGDNHostMetadataBuilder",
        )
        modules = {
            "vllm.models.qwen4_exp.amd.qsa": NS(Qwen4ExpQSAMetadataBuilder=builder),
            "vllm.models.qwen4_exp.common.qsa_cache": NS(
                QSAMetadataBuilder=state_builder, QSAStateBackend=type("State", (), {})
            ),
            "vllm.v1.attention.backends.short_conv_attn": NS(
                PleShortConvAttentionBackend=type("PLE", (), {}),
                PleShortConvAttentionMetadataBuilder=type("PLEBuilder", (), {}),
            ),
            "vllm_ascend.models.qwen4_exp": NS(qsa=NS()),
            "vllm_ascend.models.qwen4_exp.model_state": state_module,
            "vllm_ascend.models.qwen4_exp.qsa": NS(AscendQSABackend=backend, AscendQSAImpl=impl),
            "vllm_ascend.ops.gdn_attn_builder": NS(**{name: type(name, (), {}) for name in names}),
            "vllm_ascend.worker.v2.spec_decode.mtp.speculator": NS(AscendMTPSpeculator=speculator_type),
        }
        group = NS(backend=backend, get_metadata_builder=lambda _: builder())
        draft = speculator_type()
        draft.attn_groups, draft.attn_architecture, draft.attn_backend = [[group]], "QSA", backend
        runner = NS(
            model_state=state_module.AscendQwen4ExpModelState(),
            speculator=draft,
            model_config=NS(architecture="Qwen4ExpForCausalLM"),
            attn_groups=[[group]],
        )
        capability = QWEN.QwenDecodeQueueCapability()
        with patch.dict(sys.modules, modules):
            self.assertTrue(capability.supports(runner))
            draft.attn_groups = []
            self.assertFalse(capability.supports(runner))
            draft.attn_groups = [[NS(backend=backend, get_metadata_builder=lambda _: object())]]
            self.assertFalse(capability.supports(runner))
            draft.attn_groups = [[group]]
            runner.model_state = NS()
            self.assertFalse(capability.supports(runner))

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
