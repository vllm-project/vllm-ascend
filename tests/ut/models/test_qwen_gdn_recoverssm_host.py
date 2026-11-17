"""CPU host-contract tests; run directly with unittest without the NPU stack."""

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config(**overrides):
    ns = types.SimpleNamespace
    values = dict(
        additional_config={"qwen_gdn_recoverssm": True},
        model_config=ns(hf_text_config=ns(model_type="qwen4_exp_text")),
        speculative_config=ns(num_speculative_tokens=3, enable_adaptive_verification=False),
        cache_config=ns(enable_prefix_caching=False, mamba_cache_mode="none", use_replayssm=False),
        parallel_config=ns(pipeline_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1),
        use_v2_model_runner=True,
    )
    values.update(overrides)
    return ns(**values)


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_module = load_file("qwen_gdn_config_test", "vllm_ascend/models/qwen4_exp/recoverssm_config.py")

    def test_explicit_opt_in_preserves_kda_and_replayssm_identity(self):
        cfg = config()
        self.assertTrue(self.config_module.use_qwen_gdn_recoverssm(cfg))
        self.config_module.validate_qwen_gdn_recoverssm(cfg)
        self.assertFalse(cfg.cache_config.use_replayssm)
        self.assertFalse(hasattr(cfg.cache_config, "use_kda_recoverssm"))
        self.assertFalse(self.config_module.use_qwen_gdn_recoverssm(config(additional_config={})))

    def test_rejects_prefix_cache_and_non_v2_runner(self):
        cfg = config()
        cfg.cache_config.enable_prefix_caching = True
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.config_module.validate_qwen_gdn_recoverssm(cfg)
        with self.assertRaisesRegex(ValueError, "V2"):
            self.config_module.validate_qwen_gdn_recoverssm(config(use_v2_model_runner=False))

    def test_rejects_non_qwen_and_missing_speculation(self):
        cfg = config(speculative_config=None)
        with self.assertRaisesRegex(ValueError, "speculative"):
            self.config_module.validate_qwen_gdn_recoverssm(cfg)
        cfg = config()
        cfg.model_config.hf_text_config.model_type = "kimi_k3"
        with self.assertRaisesRegex(ValueError, "Qwen4Exp"):
            self.config_module.validate_qwen_gdn_recoverssm(cfg)

    def test_rejects_missing_text_model_configuration(self):
        for model_config in (
            types.SimpleNamespace(),
            types.SimpleNamespace(hf_text_config=None),
            types.SimpleNamespace(hf_text_config=types.SimpleNamespace()),
        ):
            with self.subTest(model_config=model_config), self.assertRaisesRegex(ValueError, "Qwen4Exp"):
                self.config_module.validate_qwen_gdn_recoverssm(config(model_config=model_config))

    def test_rejects_context_and_pipeline_parallelism(self):
        for field in ("pipeline_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size"):
            cfg = config()
            setattr(cfg.parallel_config, field, 2)
            with self.assertRaisesRegex(ValueError, "PP=1.*DCP=1.*PCP=1"):
                self.config_module.validate_qwen_gdn_recoverssm(cfg)

    def test_rejects_microbatch_contexts_sharing_layer_replay(self):
        cfg = config()
        cfg.parallel_config.use_ubatching = True
        with self.assertRaisesRegex(ValueError, "microbatch"):
            self.config_module.validate_qwen_gdn_recoverssm(cfg)


class CommitHostTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.layers = []
        self.counts = []

        def prepare(
            accepted,
            indices,
            starts,
            lens,
            state_lens,
            boundary_lens,
            final,
            boundary,
            recovery,
            *,
            request_indices=None,
            spec_query_len,
            **kwargs,
        ):
            counts = accepted if request_indices is None else accepted[request_indices.long()]
            lens.copy_(torch.minimum(counts.clamp_min(0), starts.diff()).clamp_max(spec_query_len))
            lens.masked_fill_(indices <= 0, 0)
            final.copy_(indices)
            boundary.zero_()
            recovery.zero_()
            self.counts.append(lens.clone())

        def commit(state, replay, table, indices, lens, final, boundary, recovery, **kwargs):
            for layer in self.layers:
                for row, page in enumerate(indices.tolist()):
                    if page <= 0:
                        continue
                    for step in range(int(lens[row])):
                        record = layer._qwen_gdn_replay[page, :, step]
                        u, key, decay = record[:, :2], record[:, 2:4], record[:, 4]
                        layer.kv_cache[1][page].mul_(decay[:, None, None]).add_(u[:, :, None] * key[:, None, :])

        def compact(
            conv,
            bases,
            block_strides,
            dim_strides,
            token_strides,
            indices,
            lens,
            final,
            boundary,
            recovery,
            *,
            conv_history_len,
            **kwargs,
        ):
            for layer in self.layers:
                for row, page in enumerate(indices.tolist()):
                    length = int(lens[row])
                    if page > 0 and length:
                        source = layer.kv_cache[0][page, :, length - 1 : length - 1 + conv_history_len].clone()
                        layer.kv_cache[0][page, :, :conv_history_len].copy_(source)

        stubs = {
            "vllm.model_executor.layers.mamba.mamba_utils": types.SimpleNamespace(is_conv_state_dim_first=lambda: True),
            "vllm_ascend.ops.triton.triton_utils": types.SimpleNamespace(init_device_properties_triton=lambda: None),
            "vllm_ascend.ops.triton.gdn.gdn_recoverssm_commit": types.SimpleNamespace(gdn_recoverssm_commit=commit),
            "vllm_ascend.ops.triton.mamba.prepare_commit_plan": types.SimpleNamespace(prepare_commit_plan=prepare),
            "vllm_ascend.ops.triton.mamba.compact_conv_state": types.SimpleNamespace(compact_conv_state=compact),
        }
        with patch.dict(sys.modules, stubs):
            self.runtime = load_file("qwen_gdn_runtime_test", "vllm_ascend/attention/recoverssm_gdn.py")

    def test_partial_acceptance_commits_only_prefix_and_canonical_conv_history(self):
        torch = self.torch
        for _ in range(2):
            conv = torch.arange(4 * 3 * 6, dtype=torch.float32).reshape(4, 3, 6)
            self.layers.append(types.SimpleNamespace(kv_cache=(conv, torch.zeros(4, 1, 2, 2))))
        context = self.runtime.GDNRecoverSSMCommitContext.create(self.layers, spec_query_len=4, max_num_reqs=3)
        for layer in self.layers:
            layer._qwen_gdn_replay.fill_(1)
            layer._qwen_gdn_replay[1, :, 2:, :2].fill_(1000)  # Rejected tail must never affect the checkpoint.
        originals = [layer.kv_cache[0].clone() for layer in self.layers]
        # Compact rows correspond to sampler rows 2 and 0, respectively.
        context.commit(
            torch.tensor([4, 1, 2]),
            torch.tensor([1, 2, 0]),
            torch.tensor([0, 4, 8, 8]),
            request_indices=torch.tensor([2, 0, 1]),
        )
        self.assertEqual(self.counts[0].tolist(), [2, 4, 0])
        for layer, original in zip(self.layers, originals):
            torch.testing.assert_close(layer.kv_cache[1][1], torch.full((1, 2, 2), 2.0))
            torch.testing.assert_close(layer.kv_cache[1][2], torch.full((1, 2, 2), 4.0))
            torch.testing.assert_close(layer.kv_cache[1][0], torch.zeros(1, 2, 2))
            torch.testing.assert_close(layer.kv_cache[0][1, :, :3], original[1, :, 1:4])
            torch.testing.assert_close(layer.kv_cache[0][2, :, :3], original[2, :, 3:6])
            torch.testing.assert_close(layer.kv_cache[0][0], original[0])

    def test_rejects_stale_cache_bindings(self):
        torch = self.torch
        layer = types.SimpleNamespace(kv_cache=(torch.zeros(2, 3, 6), torch.zeros(2, 1, 2, 2)))
        self.layers.append(layer)
        context = self.runtime.GDNRecoverSSMCommitContext.create(self.layers, spec_query_len=4, max_num_reqs=1)
        layer.kv_cache = (layer.kv_cache[0], layer.kv_cache[1].clone())
        with self.assertRaisesRegex(RuntimeError, "rebound"):
            context.commit(torch.ones(1, dtype=torch.int32), torch.tensor([1]), torch.tensor([0, 1]))


class MetadataTests(unittest.TestCase):
    def test_spec_commit_uses_sampler_mapping_and_neutral_conv_bias(self):
        import torch

        @dataclass
        class Metadata:
            num_spec_decodes: int
            spec_state_indices_tensor: object
            spec_query_start_loc: object
            spec_sequence_masks_cpu: object
            num_accepted_tokens: object

        class Recover:
            pass

        class Builder:
            def _finalize_metadata(self, metadata, fused=False):
                return metadata

        calls = []
        context = types.SimpleNamespace(commit=lambda *args, **kwargs: calls.append((args, kwargs)), commit_steps=0)
        stubs = {
            "vllm.v1.attention.backends.gdn_attn": types.SimpleNamespace(GDNAttentionMetadata=Metadata),
            "vllm.v1.attention.backends.recoverssm_metadata": types.SimpleNamespace(RecoverSSMMetadata=Recover),
            "vllm_ascend.ops.gdn_attn_builder": types.SimpleNamespace(AscendGDNHostMetadataBuilder=Builder),
            "vllm_ascend.attention.recoverssm_gdn": types.SimpleNamespace(GDNRecoverSSMCommitContext=object),
            "vllm_ascend.models.qwen4_exp.recoverssm_config": types.SimpleNamespace(
                validate_qwen_gdn_recoverssm=lambda _: None
            ),
        }
        with patch.dict(sys.modules, stubs):
            mod = load_file("qwen_gdn_metadata_test", "vllm_ascend/models/qwen4_exp/recoverssm_metadata.py")
        builder = mod.AscendGDNRecoverSSMMetadataBuilder.__new__(mod.AscendGDNRecoverSSMMetadataBuilder)
        builder._commit_context = context
        builder._neutral_accepted = torch.ones(3, dtype=torch.int32)
        metadata = Metadata(
            2,
            torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]]),
            torch.tensor([0, 4, 8]),
            torch.tensor([False, True, True]),
            torch.tensor([3, 2]),
        )
        metadata.spec_decode_metadata = types.SimpleNamespace(
            spec_causal_conv1d=types.SimpleNamespace(num_accepted_tokens=metadata.num_accepted_tokens),
            use_upstream_conv=False,
        )
        result = builder._finalize_metadata(metadata)
        self.assertIsInstance(result, Recover)
        self.assertEqual(result.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens.tolist(), [1, 1])
        self.assertEqual(metadata.num_accepted_tokens.tolist(), [3, 2])  # PLE still needs these original counts.
        sampled = torch.tensor([1, 2, 4])
        result.commit_recoverssm_state(sampled)
        self.assertIs(calls[0][0][0], sampled)
        self.assertEqual(calls[0][0][1].tolist(), [3, 7])
        self.assertEqual(calls[0][1]["request_indices"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()


def test_recover_opt_in_is_accepted_by_real_ascend_config():
    from vllm_ascend.ascend_config import AscendConfig

    assert AscendConfig(sparse_kv_offload_config=None).qwen_gdn_recoverssm is False
    assert AscendConfig(sparse_kv_offload_config=None, qwen_gdn_recoverssm=True).qwen_gdn_recoverssm is True
