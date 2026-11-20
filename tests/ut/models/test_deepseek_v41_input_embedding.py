# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_ascend.models.deepseek_v41 import input_embedding


def _device_embedding():
    layer = nn.Embedding(8, 4, dtype=torch.bfloat16)
    layer.quant_method = input_embedding.UnquantizedEmbeddingMethod()
    return layer


def _host_method(layer):
    method = Mock(metadata=torch.empty_like(layer.weight, device="meta"))
    method.embedding.side_effect = layer.quant_method.embedding
    return method


def _rank_local_embedding(address):
    embedding = nn.Module()
    embedding.quant_method = input_embedding.UnquantizedEmbeddingMethod()
    embedding.weight = Mock(device=Mock(type="npu"), dtype=torch.bfloat16, ndim=2)
    embedding.weight.numel.return_value = 32
    embedding.weight.is_contiguous.return_value = True
    embedding.weight.untyped_storage.return_value.data_ptr.return_value = address
    return embedding


class TestHostVmmInputEmbedding(TestCase):
    def test_selects_independent_tables_and_deduplicates_shared_module(self):
        target, draft, head = (_rank_local_embedding(address) for address in (1, 2, 1))
        head.weight.device = target.weight.device
        target_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=target), lm_head=head)
        draft_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=draft))
        with patch.object(input_embedding, "VocabParallelEmbedding", nn.Module):
            self.assertEqual(input_embedding.get_input_embedding_layers(target_model, None), [])
            self.assertEqual(input_embedding.get_input_embedding_layers(target_model, draft_model), [draft])
            target_model.model.embed_tokens = draft
            self.assertEqual(input_embedding.get_input_embedding_layers(target_model, draft_model), [draft])

    def test_selects_only_embeddings_resident_on_the_pipeline_stage(self):
        target, draft = _rank_local_embedding(1), _rank_local_embedding(2)
        target_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=target), lm_head=PPMissingLayer())
        draft_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=draft))
        with patch.object(input_embedding, "VocabParallelEmbedding", nn.Module):
            self.assertEqual(input_embedding.get_input_embedding_layers(target_model, None), [target])
            target_model.model.embed_tokens = PPMissingLayer()
            self.assertEqual(input_embedding.get_input_embedding_layers(target_model, None), [])
            self.assertEqual(input_embedding.get_input_embedding_layers(target_model, draft_model), [draft])

    def test_prepares_both_tables_before_releasing_device_weights(self):
        target, draft = _device_embedding(), _device_embedding()
        methods = [_host_method(layer) for layer in (target, draft)]

        with patch.object(input_embedding, "HostVmmEmbeddingMethod", side_effect=methods):
            self.assertEqual(input_embedding.offload_input_embeddings([target, draft]), methods)

        for layer, method in zip((target, draft), methods):
            self.assertEqual(layer.weight.device.type, "meta")
            self.assertIs(layer.quant_method, method)

    def test_second_mapping_failure_leaves_both_device_weights_installed(self):
        target, draft = _device_embedding(), _device_embedding()
        target_weight, draft_weight = target.weight, draft.weight

        with (
            patch.object(
                input_embedding, "HostVmmEmbeddingMethod", side_effect=[_host_method(target), RuntimeError("mapping")]
            ),
            self.assertRaisesRegex(RuntimeError, "mapping"),
        ):
            input_embedding.offload_input_embeddings([target, draft])

        self.assertIs(target.weight, target_weight)
        self.assertIs(draft.weight, draft_weight)

    def test_lookup_mismatch_leaves_the_device_weight_installed(self):
        layer = _device_embedding()
        weight = layer.weight
        method = _host_method(layer)
        method.embedding.side_effect = lambda layer, ids: layer.quant_method.embedding(layer, ids) + 1

        with (
            patch.object(input_embedding, "HostVmmEmbeddingMethod", return_value=method),
            self.assertRaisesRegex(RuntimeError, "differs from the loaded weight"),
        ):
            input_embedding.offload_input_embeddings([layer])

        self.assertIs(layer.weight, weight)

    def test_failed_release_preserves_mapping_for_retry(self):
        method = input_embedding.HostVmmEmbeddingMethod.__new__(input_embedding.HostVmmEmbeddingMethod)
        mapping = Mock()
        mapping.close.side_effect = [RuntimeError("unmap"), None]
        directory = Mock()
        method.mapping = mapping
        method.directory = directory
        method.ptrs = Mock()

        with self.assertRaisesRegex(RuntimeError, "unmap"):
            method.close_host_offload()
        self.assertIs(method.mapping, mapping)
        directory.cleanup.assert_not_called()

        method.close_host_offload()
        method.close_host_offload()
        self.assertEqual(mapping.close.call_count, 2)
        directory.cleanup.assert_called_once()
        self.assertIsNone(method.ptrs)

    def test_graph_metadata_keeps_input_shape_and_bf16_dtype(self):
        with FakeTensorMode():
            ptrs = torch.empty(1, dtype=torch.int64)
            for shape in ((0,), (3, 2)):
                ids = torch.empty(shape, dtype=torch.int64)
                output = input_embedding._host_vmm_lookup(ids, ptrs, 8, 4)
                self.assertEqual(output.shape, (*shape, 4))
                self.assertEqual(output.dtype, torch.bfloat16)
