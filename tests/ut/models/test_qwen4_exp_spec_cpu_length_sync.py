# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for Qwen's lifetime-stable speculative length policy."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "vllm_ascend/models/qwen4_exp"
CONFIG_KEY = "enable_decode_early_queue"


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


def config_module() -> tuple[Any, dict[str, Any]]:
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
    spec = importlib.util.spec_from_file_location("_sync_test_config", ROOT / "vllm_ascend/ascend_config.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    modules[spec.name] = module
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, modules


class TestSpecCpuLengthSync(unittest.TestCase):
    def setUp(self):
        bases = {
            "vllm.models.qwen4_exp.amd.model_state": NS(Qwen4ExpModelState=type("QwenState", (), {})),
            "vllm_ascend.worker.v2.model_states.mamba_hybrid": NS(
                AscendMambaHybridModelState=type("HybridState", (), {})
            ),
        }
        with patch.dict(sys.modules, bases):
            self.state_module = load("_sync_test_state", MODEL / "model_state.py")
        self.state_type = self.state_module.AscendQwen4ExpModelState
        self.builder = type("QSABuilder", (), {})
        self.state_builder = type("QSAStateBuilder", (), {})
        self.impl = type("QSAImpl", (), {})
        self.backend = type("QSABackend", (), {"get_impl_cls": staticmethod(lambda: self.impl)})
        self.speculator_type = type("MTPSpeculator", (), {})
        qsa = NS(AscendQSABackend=self.backend, AscendQSAImpl=self.impl)
        self.modules = {
            **bases,
            "vllm.models.qwen4_exp.amd.qsa": NS(Qwen4ExpQSAMetadataBuilder=self.builder),
            "vllm.models.qwen4_exp.common.qsa_cache": NS(
                QSAMetadataBuilder=self.state_builder, QSAStateBackend=type("QSAStateBackend", (), {})
            ),
            "vllm.v1.attention.backends.short_conv_attn": NS(
                PleShortConvAttentionBackend=type("PLEBackend", (), {}),
                PleShortConvAttentionMetadataBuilder=type("PLEBuilder", (), {}),
            ),
            "vllm_ascend.models.qwen4_exp": NS(qsa=qsa),
            "vllm_ascend.models.qwen4_exp.model_state": self.state_module,
            "vllm_ascend.models.qwen4_exp.qsa": qsa,
            "vllm_ascend.ops.gdn_attn_builder": NS(
                **{
                    name: type(name, (), {})
                    for name in (
                        "AscendGDNAttentionBackend",
                        "AscendGDNAttentionMetadataBuilder",
                        "AscendGDNFusedAttentionBackend",
                        "AscendGDNHostMetadataBackend",
                        "AscendGDNHostMetadataBuilder",
                    )
                }
            ),
            "vllm_ascend.worker.v2.spec_decode.mtp.speculator": NS(AscendMTPSpeculator=self.speculator_type),
        }
        self.policy = load("_sync_test_policy", MODEL / "cpu_length_sync.py")
        self.modules["vllm_ascend.models.qwen4_exp.cpu_length_sync"] = self.policy
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def runner(self) -> Any:
        group = NS(backend=self.backend, get_metadata_builder=lambda _: self.builder())
        draft = self.speculator_type()
        draft.attn_groups, draft.attn_architecture, draft.attn_backend = [[group]], "QSA", self.backend
        return NS(
            model_state=self.state_type(),
            model_config=NS(architecture="Qwen4ExpForCausalLM"),
            ascend_config=NS(enable_decode_early_queue=True, sparse_kv_offload_config=NS(enabled=False)),
            parallel_config=NS(decode_context_parallel_size=1, prefill_context_parallel_size=1),
            speculator=draft,
            attn_groups=[[group]],
            num_computed_tokens_stream=NS(synchronize=Mock()),
            supports_mm_inputs=False,
        )

    def eligible(self, runner: Any) -> bool:
        return self.policy.can_skip_qwen_cpu_length_sync(runner)

    def test_additional_config_defaults_and_validates_json_boolean(self):
        module, modules = config_module()
        defaults = dict(sparse_kv_offload_config=NS(enabled=False), multistream_engram_overlap=False)
        # Keep the real complete schema available for pydantic forward references.
        with patch.dict(sys.modules, modules):
            for payload, expected in (({}, False), ({CONFIG_KEY: True}, True), ({CONFIG_KEY: "false"}, False)):
                with self.subTest(payload=payload):
                    config = module.AscendConfig(**defaults, **json.loads(json.dumps(payload)))
                    self.assertIs(getattr(config, CONFIG_KEY), expected)
            for invalid in (None, [True], "invalid"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    module.AscendConfig(**defaults, **{CONFIG_KEY: invalid})

    def test_opt_out_does_not_inspect_optional_model_state(self):
        for config in (NS(), NS(enable_decode_early_queue=False)):
            with self.subTest(config=config):
                self.assertFalse(self.eligible(NS(ascend_config=config)))

    def test_audited_target_and_draft_use_device_lengths(self):
        qsa_cache = self.modules["vllm.models.qwen4_exp.common.qsa_cache"]
        short_conv = self.modules["vllm.v1.attention.backends.short_conv_attn"]
        gdn = self.modules["vllm_ascend.ops.gdn_attn_builder"]
        pairs = (
            (qsa_cache.QSAStateBackend, self.state_builder),
            (short_conv.PleShortConvAttentionBackend, short_conv.PleShortConvAttentionMetadataBuilder),
            (gdn.AscendGDNAttentionBackend, gdn.AscendGDNAttentionMetadataBuilder),
            (gdn.AscendGDNHostMetadataBackend, gdn.AscendGDNHostMetadataBuilder),
            (gdn.AscendGDNFusedAttentionBackend, gdn.AscendGDNAttentionMetadataBuilder),
        )
        for architecture in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
            with self.subTest(architecture=architecture):
                runner = self.runner()
                runner.model_config.architecture = architecture
                for backend, builder in pairs:
                    group = NS(backend=backend, get_metadata_builder=lambda _, builder=builder: builder())
                    runner.attn_groups.append([group])
                    runner.speculator.attn_groups.append([group])
                self.assertTrue(self.eligible(runner))

    def test_other_model_or_speculator_retains_exact_sync(self):
        for changed in ("state", "architecture", "speculator"):
            with self.subTest(changed=changed):
                runner = self.runner()
                if changed == "state":
                    runner.model_state = NS()
                elif changed == "architecture":
                    runner.model_config.architecture = "LlamaForCausalLM"
                else:
                    runner.speculator = NS()
                self.assertFalse(self.eligible(runner))

    def test_unknown_target_or_draft_consumer_retains_exact_sync(self):
        for owner in ("target", "draft"):
            for groups in (
                [],
                [[NS(backend=self.backend, get_metadata_builder=lambda _: object())]],
                [[NS(backend=object, get_metadata_builder=lambda _: self.builder())]],
                [[NS(backend=self.backend, get_metadata_builder=Mock(side_effect=RuntimeError("unknown builder")))]],
            ):
                with self.subTest(owner=owner, groups=groups):
                    runner = self.runner()
                    consumer = runner if owner == "target" else runner.speculator
                    consumer.attn_groups = groups
                    self.assertFalse(self.eligible(runner))

    def test_changed_native_implementation_retains_exact_sync(self):
        runner = self.runner()
        with patch.object(self.backend, "get_impl_cls", return_value=object):
            self.assertFalse(self.eligible(runner))

    def test_context_parallel_and_sparse_cpu_consumers_retain_exact_sync(self):
        for field in ("decode_context_parallel_size", "prefill_context_parallel_size", "sparse"):
            with self.subTest(field=field):
                runner = self.runner()
                if field == "sparse":
                    runner.ascend_config.sparse_kv_offload_config.enabled = True
                else:
                    setattr(runner.parallel_config, field, 2)
                self.assertFalse(self.eligible(runner))

    def test_unrelated_parallel_and_scheduler_options_do_not_disable_policy(self):
        runner = self.runner()
        runner.parallel_config.tensor_parallel_size = 4
        runner.parallel_config.data_parallel_size = 2
        runner.parallel_config.pipeline_parallel_size = 2
        runner.parallel_config.enable_dbo = True
        runner.parallel_config.use_sequence_parallel_moe = True
        runner.cache_config = NS(enable_prefix_caching=True)
        runner.lora_config = object()
        runner.adaptive_verification = object()
        self.assertTrue(self.eligible(runner))

    def test_non_realtime_multimodal_uses_same_policy_without_media_proof(self):
        runner = self.runner()
        runner.model_config.architecture = "Qwen4ExpForConditionalGeneration"
        runner.supports_mm_inputs = True
        runner.model_state.encoder_runner = NS(is_realtime=False)
        self.assertTrue(self.eligible(runner))
        self.assertFalse(hasattr(runner.model_state, "_decode_submission"))

    def test_unknown_or_realtime_encoder_retains_exact_sync(self):
        for encoder in (None, NS(), NS(is_realtime=True)):
            with self.subTest(encoder=encoder):
                runner = self.runner()
                runner.supports_mm_inputs = True
                runner.model_state.encoder_runner = encoder
                self.assertFalse(self.eligible(runner))

    def test_missing_optional_native_schema_falls_back(self):
        runner = self.runner()
        del runner.speculator.attn_architecture
        self.assertFalse(self.eligible(runner))

    def test_uninitialized_model_state_uses_exact_sync(self):
        self.assertFalse(self.state_type().can_skip_spec_cpu_length_sync())

    def test_policy_is_frozen_until_next_kv_initialization(self):
        runner = self.runner()
        state = runner.model_state
        state.initialize_spec_cpu_length_sync(runner)
        self.assertTrue(state.can_skip_spec_cpu_length_sync())
        runner.ascend_config.enable_decode_early_queue = False
        runner.speculator.attn_groups = []
        with patch.object(self.policy, "can_skip_qwen_cpu_length_sync", side_effect=AssertionError("per-step check")):
            self.assertTrue(state.can_skip_spec_cpu_length_sync())
            self.assertTrue(state.can_skip_spec_cpu_length_sync())
        state.initialize_spec_cpu_length_sync(runner)
        self.assertFalse(state.can_skip_spec_cpu_length_sync())

    def test_disabled_state_also_stays_frozen_until_reinitialization(self):
        runner = self.runner()
        runner.ascend_config.enable_decode_early_queue = False
        state = runner.model_state
        state.initialize_spec_cpu_length_sync(runner)
        runner.ascend_config.enable_decode_early_queue = True
        self.assertFalse(state.can_skip_spec_cpu_length_sync())
        state.initialize_spec_cpu_length_sync(runner)
        self.assertTrue(state.can_skip_spec_cpu_length_sync())

    def test_pending_exact_snapshot_is_drained_before_enabling_policy(self):
        runner = self.runner()
        state = runner.model_state
        runner.num_computed_tokens_stream.synchronize.side_effect = lambda: self.assertFalse(
            state.can_skip_spec_cpu_length_sync()
        )
        state.initialize_spec_cpu_length_sync(runner)
        runner.num_computed_tokens_stream.synchronize.assert_called_once_with()
        self.assertTrue(state.can_skip_spec_cpu_length_sync())
        for supported in (False, True):
            with self.subTest(supported=supported):
                other = self.runner()
                other.ascend_config.enable_decode_early_queue = supported
                if supported:
                    other.speculator.attn_groups = []
                other.model_state.initialize_spec_cpu_length_sync(other)
                other.num_computed_tokens_stream.synchronize.assert_not_called()
                self.assertFalse(other.model_state.can_skip_spec_cpu_length_sync())
        for stream in (None, NS(synchronize=None)):
            with self.subTest(stream=stream):
                runner.num_computed_tokens_stream = stream
                state.initialize_spec_cpu_length_sync(runner)
                self.assertFalse(state.can_skip_spec_cpu_length_sync())
        runner.num_computed_tokens_stream = NS(synchronize=Mock(side_effect=RuntimeError("D2H failed")))
        with self.assertRaisesRegex(RuntimeError, "D2H failed"):
            state.initialize_spec_cpu_length_sync(runner)
        self.assertFalse(state.can_skip_spec_cpu_length_sync())


if __name__ == "__main__":
    unittest.main()
