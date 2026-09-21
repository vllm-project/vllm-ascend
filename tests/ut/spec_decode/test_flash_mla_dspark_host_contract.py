# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute DSpark Python contracts on CPU, without NPU/upstream imports.

Run directly. Selected repository methods execute unchanged; only heavyweight
construction, upstream dispatch and package dependencies are stubbed. This is
not model loading, NPU operator correctness, or graph execution evidence.
"""

import ast
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]
MODEL = "vllm_ascend/models/kimi_k3_dspark.py"
SPEC = "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"


def load_selected(path, functions=(), classes=None, **namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in (classes or {}):
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in classes[node.name]]
            selected.append(node)
    tree.body = selected
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class DSparkHostContract(unittest.TestCase):
    def model_scope(self):
        def load(owner, **kwargs):
            def consume(weights, **kwargs):
                owner.loaded = dict(weights)
                return set(owner.loaded)

            return SimpleNamespace(load_weights=consume)

        weights_scope = load_selected(
            "vllm_ascend/models/qwen3_dspark.py",
            ["process_weight", "align_draft_weights"],
            torch=torch,
            get_rotation_path=lambda cfg: getattr(cfg, "rotation", None),
            get_rotation_matrix=Mock(return_value=torch.tensor([[0.0, 1.0], [-1.0, 0.0]])),
            VocabParallelEmbedding=Mock(side_effect=lambda *a, **kw: SimpleNamespace(quant_method=Mock())),
            ParallelLMHead=Mock(side_effect=lambda *a, **kw: SimpleNamespace(quant_method=Mock())),
            load_quarot_target_layer=Mock(),
            TARGET_EMBED_WEIGHT_NAMES=("embed",),
            TARGET_LM_HEAD_WEIGHT_NAMES=("head",),
        )
        scope = load_selected(
            MODEL,
            classes={
                "AscendK3DSparkForCausalLM": {
                    "__init__",
                    "configure_target_aux_hidden_capture",
                    "load_weights",
                    "post_process",
                }
            },
            torch=torch,
            nn=torch.nn,
            UpstreamK3DSparkForCausalLM=torch.nn.Module,
            AscendK3DSparkModel=Mock(return_value=SimpleNamespace(embed_tokens=None)),
            LogitsProcessor=Mock(),
            maybe_prefix=lambda prefix, name: f"{prefix}.{name}" if prefix else name,
            AutoWeightsLoader=load,
            align_draft_weights=weights_scope.align_draft_weights,
        )
        scope.weights = weights_scope
        return scope

    @staticmethod
    def config(rotation=None):
        draft = SimpleNamespace(draft_vocab_size=8)
        return SimpleNamespace(
            rotation=rotation,
            speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=draft)),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=93, hidden_size=2, vocab_size=8),
                get_num_layers=lambda parallel: 31,
                model="target-weights",
            ),
            parallel_config=object(),
        )

    def test_constructor_keeps_main_framework_weight_ownership(self):
        scope = self.model_scope()
        model = scope.AscendK3DSparkForCausalLM(vllm_config=self.config("target"))
        self.assertEqual(scope.AscendK3DSparkModel.call_args.kwargs["start_layer_id"], 31)
        self.assertIsNone(model.model.embed_tokens)
        self.assertIsNone(model.lm_head)
        self.assertFalse(hasattr(model, "rotation_path"))

    def test_loading_preserves_weights_until_post_process(self):
        scope = self.model_scope()
        cfg = self.config("target")
        model = scope.AscendK3DSparkForCausalLM(vllm_config=cfg)
        model.hf_to_vllm_mapper = object()
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        model.load_weights(iter([("model.context_proj.weight", weight)]))
        self.assertIs(model.loaded["model.context_proj.weight"], weight)
        scope.weights.get_rotation_matrix.assert_not_called()
        model.model.context_proj = SimpleNamespace(weight=weight)
        target_embed, target_head = object(), object()
        model.model.embed_tokens, model.lm_head = target_embed, target_head
        model.post_process(cfg)
        torch.testing.assert_close(weight, torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))
        scope.weights.get_rotation_matrix.assert_called_once_with("target")
        self.assertIsNot(model.model.embed_tokens, target_embed)
        self.assertIsNot(model.lm_head, target_head)
        self.assertTrue(model.has_own_embed_tokens)
        self.assertTrue(model.has_own_lm_head)

    def test_unrotated_post_process_preserves_shared_weights(self):
        scope = self.model_scope()
        cfg = self.config()
        model = scope.AscendK3DSparkForCausalLM(vllm_config=cfg)
        weight = torch.ones(1, 4)
        model.model.context_proj = SimpleNamespace(weight=weight)
        target_embed, target_head = object(), object()
        model.model.embed_tokens, model.lm_head = target_embed, target_head
        model.post_process(cfg)
        self.assertIs(model.model.embed_tokens, target_embed)
        self.assertIs(model.lm_head, target_head)
        scope.weights.get_rotation_matrix.assert_not_called()

    def test_draft_loader_uses_main_post_process_once_after_upstream(self):
        calls = []
        model = SimpleNamespace(
            post_process=lambda cfg: calls.append("post"),
            configure_target_aux_hidden_capture=lambda target: calls.append("aux"),
        )

        class Upstream:
            def load_draft_model(self, target, names):
                calls.append("load-and-share")
                return model

        scope = load_selected(
            SPEC,
            classes={"AscendDSparkSpeculator": {"load_draft_model"}},
            DSparkSpeculator=Upstream,
            set_current_vllm_config=lambda cfg: nullcontext(),
        )
        spec = scope.AscendDSparkSpeculator()
        spec.vllm_config = object()
        self.assertIs(spec.load_draft_model(object(), set()), model)
        self.assertEqual(calls, ["load-and-share", "post", "aux"])

    def aux_pair(self):
        model = SimpleNamespace(
            config=SimpleNamespace(target_layer_ids=[0, 2], target_hidden_size=4, num_target_layers=2)
        )
        target = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(num_hidden_layers=4, hidden_size=4), aux_hidden_state_layers=(1, 3)
            ),
            set_dspark_aux_capture_materialized=Mock(),
        )
        return model, target

    def test_aux_raw_capture_and_multimodal_wrapper(self):
        hook = self.model_scope().AscendK3DSparkForCausalLM.configure_target_aux_hidden_capture
        for wrapped in (False, True):
            model, target = self.aux_pair()
            hook(model, SimpleNamespace(get_language_model=lambda target=target: target) if wrapped else target)
            target.set_dspark_aux_capture_materialized.assert_called_once_with(False)

    def test_aux_rejects_bad_contract_before_setting_mode(self):
        hook = self.model_scope().AscendK3DSparkForCausalLM.configure_target_aux_hidden_capture
        for field, value in [
            ("target_layer_ids", []),
            ("target_layer_ids", [0, 0]),
            ("target_layer_ids", [-1, 2]),
            ("target_layer_ids", [0, 4]),
            ("target_layer_ids", [1, 2]),
            ("target_hidden_size", 8),
            ("num_target_layers", 3),
        ]:
            with self.subTest(field=field, value=value):
                model, target = self.aux_pair()
                setattr(model.config, field, value)
                with self.assertRaises(ValueError):
                    hook(model, target)
                target.set_dspark_aux_capture_materialized.assert_not_called()
        model, target = self.aux_pair()
        del target.set_dspark_aux_capture_materialized
        with self.assertRaises(ValueError):
            hook(model, target)

    def spec_scope(self, architecture="MLA", flash=True):
        class Upstream:
            def _build_draft_attn_metadata(self, **kwargs):
                self.calls.append(kwargs)
                return self.metadata

            def set_attn(self, *args):
                pass

        contexts = []

        @contextmanager
        def factory(positions, pad, is_prefilling, **kwargs):
            contexts.append((pad, is_prefilling.clone(), kwargs))
            yield

        scope = load_selected(
            SPEC,
            classes={
                "AscendDSparkSpeculator": {
                    "set_attn",
                    "_draft_query_attn_state",
                    "draft_capture_context",
                    "_build_draft_attn_metadata",
                    "build_draft_attn_metadatas",
                    "_update_draft_attn_metadata",
                }
            },
            DSparkSpeculator=Upstream,
            torch=torch,
            contextmanager=contextmanager,
            ascend_envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=flash),
            AscendAttentionState=SimpleNamespace(SpecDecoding="spec", ChunkedPrefill="chunked"),
            build_attn_metadata_wrapper=nullcontext,
            build_draft_attn_metadata_factory=factory,
            dflash_cudagraph=SimpleNamespace(build_attn_metadata=Mock()),
            build_attn_metadata=Mock(return_value={}),
            set_current_vllm_config=lambda cfg: nullcontext(),
            _get_graph_update_backend=lambda groups: groups,
            AscendMLABackend=type("MLA", (), {}),
            AscendAttentionBackend=type("GQA", (), {}),
        )
        spec = scope.AscendDSparkSpeculator()
        spec.attn_architecture = architecture
        spec.num_query_per_req = 5
        spec.input_buffers = SimpleNamespace(positions=torch.arange(32))
        spec.input_batch = SimpleNamespace(num_reqs=1)
        spec._group_causal = {0: False}
        spec.calls = []
        spec.metadata = {}
        spec.contexts = contexts
        return scope, spec

    def test_backend_selection_uses_draft_not_target_configuration(self):
        scope, spec = self.spec_scope()
        spec.attn_vllm_config = object()
        spec._context_slot_mappings = torch.arange(4)
        spec.draft_attn_layer_names = set()
        for backend, expected in [
            (scope.AscendMLABackend, "MLA"),
            (scope.AscendAttentionBackend, "GQA"),
            (object, None),
        ]:
            spec.attn_groups = backend
            spec.set_attn(None, SimpleNamespace(kv_cache_groups=[]), None, None, None)
            self.assertEqual(spec.attn_architecture, expected)
            self.assertEqual(spec._context_slot_mappings.dtype, torch.int32)

    def test_fia_dp_padding_and_state(self):
        for architecture in ("MLA", "GQA"):
            with self.subTest(architecture=architecture):
                _, spec = self.spec_scope(architecture, flash=False)
                query = SimpleNamespace(actual_seq_lengths_q=[5])
                spec.metadata = {
                    "draft": SimpleNamespace(decode=query, attn_state="chunked") if architecture == "MLA" else query
                }
                spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=1, num_tokens_padded=10, step=5)
                self.assertEqual(spec.calls[0]["num_reqs_padded"], 2)
                self.assertEqual(query.actual_seq_lengths_q, [5, 10])
                pad, flags, kwargs = spec.contexts[0]
                self.assertEqual(pad, 10)
                self.assertEqual(flags.tolist(), [False, False])
                self.assertEqual(kwargs["attn_state"], "chunked")
                with self.assertRaisesRegex(AssertionError, "whole query groups"):
                    spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=1, num_tokens_padded=9)
                self.assertEqual(len(spec.calls), 1)

    def test_flash_keeps_device_boundaries_and_zero_used_padding(self):
        _, spec = self.spec_scope()
        flash = SimpleNamespace(cu=torch.tensor([0, 5, 12]), used_q=torch.tensor([5, 0]))
        spec.metadata = {"draft": SimpleNamespace(flash=flash, decode=None)}
        result = spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=1, num_tokens_padded=12, step=5)
        self.assertIs(result["draft"].flash, flash)
        self.assertEqual(spec.calls[0]["num_reqs_padded"], 1)
        self.assertEqual(flash.cu.tolist(), [0, 5, 12])
        self.assertEqual(flash.used_q.tolist(), [5, 0])
        self.assertEqual(spec.contexts[0][2]["attn_state"], "spec")

    def test_full_builder_calls_parent_once_and_keeps_group_causality(self):
        _, spec = self.spec_scope()
        spec.metadata = {"draft": SimpleNamespace(flash=object(), decode=None)}
        self.assertEqual(spec.build_draft_attn_metadatas(4, torch.tensor([20])), [spec.metadata])
        self.assertEqual(len(spec.calls), 1)
        self.assertEqual(spec.calls[0]["num_tokens_padded"], 20)
        self.assertEqual(spec.calls[0]["causal"], {0: False})
        self.assertEqual(spec.contexts[0][1].tolist(), [False] * 4)

    def test_sparse_metadata_is_not_normalized_as_dense(self):
        _, spec = self.spec_scope(None)
        spec.metadata = {"sparse": SimpleNamespace(actual_seq_lengths_q=[5, 5])}
        self.assertIs(spec._build_draft_attn_metadata(num_reqs_padded=2), spec.metadata)
        self.assertEqual(spec.metadata["sparse"].actual_seq_lengths_q, [5, 5])
        self.assertEqual(spec.contexts, [])

    def test_capture_factory_matches_runtime_and_restores_on_failure(self):
        for flash in (False, True):
            with self.subTest(flash=flash):
                scope, spec = self.spec_scope(flash=flash)
                original = scope.dflash_cudagraph.build_attn_metadata
                with self.assertRaisesRegex(RuntimeError, "capture failed"), spec.draft_capture_context():
                    scope.dflash_cudagraph.build_attn_metadata(num_tokens=10, num_reqs=2)
                    kwargs = scope.build_attn_metadata.call_args.kwargs
                    self.assertEqual(kwargs["attn_state"], spec._draft_query_attn_state())
                    self.assertEqual(kwargs["positions"].numel(), 10)
                    self.assertEqual(kwargs["is_prefilling"].tolist(), [False, False])
                    raise RuntimeError("capture failed")
                self.assertIs(scope.dflash_cudagraph.build_attn_metadata, original)

    def test_context_precompute_uses_each_draft_layers_original_slots(self):
        calls = []

        class Upstream:
            pass

        scope = load_selected(
            MODEL,
            classes={"AscendK3DSparkModel": {"precompute_and_store_context_kv"}},
            UpstreamK3DSparkModel=Upstream,
            torch=torch,
            get_cos_and_sin_mla=lambda positions: (positions, positions),
        )

        def make_layer():
            def writer(kv, cos, sin, cache, slots):
                calls.append((kv, cos, cache, slots))

            return SimpleNamespace(
                self_attn=SimpleNamespace(
                    fused_qkv_a_proj=lambda x: (torch.ones(x.shape[0], 578),),
                    q_lora_rank=2,
                    kv_cache=object(),
                    impl=SimpleNamespace(exec_kv_prefill=writer),
                )
            )

        model = scope.AscendK3DSparkModel()
        model.layers = [make_layer(), make_layer()]
        slots = [torch.tensor([127, 128, -1]), torch.tensor([255, -1, 256])]
        positions, states = torch.arange(3), torch.ones(3, 4)
        model.precompute_and_store_context_kv(states, positions, slots)
        self.assertEqual(len(calls), 2)
        for index, (kv, actual_positions, cache, actual_slots) in enumerate(calls):
            self.assertEqual(kv.shape, (3, 576))
            self.assertIs(actual_positions, positions)
            self.assertIs(cache, model.layers[index].self_attn.kv_cache)
            self.assertIs(actual_slots, slots[index])
        model.precompute_and_store_context_kv(states, positions, None)
        model.precompute_and_store_context_kv(states[:0], positions[:0], slots)
        self.assertEqual(len(calls), 2)

    def graph_scope(self, architecture):
        class Upstream:
            def run_fullgraph(self, desc):
                self.replays += 1
                return "replayed"

        scope = load_selected(
            "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py",
            classes={"DFlashAclGraphManager": {"run_fullgraph"}},
            DFlashCudaGraphManager=Upstream,
            ascend_envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            use_updatable_graph=lambda backend: True,
        )
        manager = scope.DFlashAclGraphManager()
        manager.replays = 0
        manager.speculator = SimpleNamespace(
            attn_architecture=architecture,
            attn_backends={"draft": object()},
            build_draft_attn_metadatas=Mock(return_value=[{}]),
            input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
        )
        manager._updatable_graph_replay = Mock(return_value="updated")
        return manager

    def test_flash_full_replay_does_not_rebuild_or_resubmit_metadata(self):
        manager = self.graph_scope("MLA")
        self.assertEqual(manager.run_fullgraph(SimpleNamespace(num_tokens=10, num_reqs=2)), "replayed")
        self.assertEqual(manager.replays, 1)
        manager.speculator.build_draft_attn_metadatas.assert_not_called()
        manager._updatable_graph_replay.assert_not_called()

    def test_gqa_keeps_main_updatable_graph_route_with_flash_enabled(self):
        manager = self.graph_scope("GQA")
        self.assertEqual(manager.run_fullgraph(SimpleNamespace(num_tokens=10, num_reqs=2)), "updated")
        self.assertEqual(manager.replays, 0)
        manager.speculator.build_draft_attn_metadatas.assert_called_once()
        manager._updatable_graph_replay.assert_called_once()


if __name__ == "__main__":
    unittest.main()
