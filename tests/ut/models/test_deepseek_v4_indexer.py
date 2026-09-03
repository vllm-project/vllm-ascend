# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata
from vllm_ascend.models.deepseek_v4.indexer import (
    AscendIndexerMetadata,
    AscendIndexerOps,
    DeepseekV4Indexer,
    IndexerOverlapPlan,
    hadamard_linear,
    hadamard_scale,
    rotate_activation,
)


def _make_indexer(topk_indices_buffer: torch.Tensor | None) -> DeepseekV4Indexer:
    indexer = DeepseekV4Indexer.__new__(DeepseekV4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.topk_indices_buffer = topk_indices_buffer
    return indexer


class TestHadamardTransform:
    def test_linear_pads_and_scale_restores_original_shape(self):
        value = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        hadamard = torch.eye(4)

        projected, original_shape, original_dim = hadamard_linear(value, hadamard)
        actual = hadamard_scale(
            projected,
            original_shape,
            original_dim,
            scale=2.0,
        )

        assert projected.shape == (2, 4)
        assert original_shape == value.shape
        assert original_dim == 3
        assert torch.equal(actual, value * 2)

    def test_rotate_activation_uses_original_dimension_scale(self):
        value = torch.tensor([[1.0, 2.0, 3.0]])

        actual = rotate_activation(value, torch.eye(4))

        torch.testing.assert_close(actual, value * (3**-0.5))


class TestIndexerMetadata:
    def test_returns_cache_request_metadata_and_hadamard(self):
        request_metadata = object()
        hadamard = torch.eye(4)
        metadata = AscendIndexerMetadata(
            compressor=AscendCompressorMetadata(
                cache=SimpleNamespace(
                    req_metadata=request_metadata,
                    hadamard=hadamard,
                ),
                state=object(),
            )
        )

        actual_metadata, actual_hadamard = DeepseekV4Indexer._get_indexer_cache_metadata(metadata)

        assert actual_metadata is request_metadata
        assert actual_hadamard is hadamard


class TestIndexerTopKCache:
    def test_update_and_read_respect_offset_and_singleton_head(self):
        buffer = torch.full((4, 3), -1, dtype=torch.int32)
        indexer = _make_indexer(buffer)
        topk_indices = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]], dtype=torch.int32)

        indexer._update_cached_topk_indices(topk_indices, offset=1)
        actual = indexer._get_cached_topk_indices(num_tokens=2, offset=1)

        assert actual.shape == (2, 1, 3)
        assert torch.equal(actual, topk_indices)
        assert torch.equal(buffer[0], torch.full((3,), -1, dtype=torch.int32))
        assert torch.equal(buffer[3], torch.full((3,), -1, dtype=torch.int32))

    def test_read_requires_buffer(self):
        indexer = _make_indexer(None)

        with pytest.raises(RuntimeError, match="topk_indices_buffer is required"):
            indexer._get_cached_topk_indices(num_tokens=1)

    def test_update_rejects_non_singleton_head_dimension(self):
        indexer = _make_indexer(torch.full((2, 3), -1, dtype=torch.int32))
        topk_indices = torch.ones((2, 2, 3), dtype=torch.int32)

        with pytest.raises(ValueError, match="singleton head dimension"):
            indexer._update_cached_topk_indices(topk_indices)


def _make_forward_metadata() -> tuple[AscendIndexerMetadata, torch.Tensor, torch.Tensor]:
    cos = torch.arange(4).view(2, 1, 1, 2)
    sin = -cos
    request_metadata = SimpleNamespace(cos={"layer": cos}, sin={"layer": sin})
    metadata = AscendIndexerMetadata(
        compressor=AscendCompressorMetadata(
            cache=SimpleNamespace(
                req_metadata=request_metadata,
                hadamard=torch.eye(4),
            ),
            state=object(),
        )
    )
    return metadata, cos, sin


class TestIndexerCacheUpdate:
    def test_updates_compressor_and_quantized_key_caches(self):
        indexer = _make_indexer(None)
        metadata, _, _ = _make_forward_metadata()
        hidden_states = torch.ones((2, 4))
        state_cache = torch.empty(0)
        key_cache = torch.empty(0)
        scale_cache = torch.empty(0)
        full_cache = torch.empty(0)
        kv_cache = (torch.empty(0),)
        key = torch.ones((2, 1, 4))
        rotated_key = torch.full_like(key, 2)
        key_scale = torch.ones((2, 1))
        slot_mapping = torch.zeros((2, 2), dtype=torch.int32)
        compressor = MagicMock(return_value=(key, slot_mapping))
        compressor.rotate = True
        indexer.compressor = compressor
        indexer.ops = SimpleNamespace(
            unpack_dsa_indexer_kv_cache=MagicMock(
                return_value=(
                    state_cache,
                    key_cache,
                    scale_cache,
                    full_cache,
                )
            ),
            quantize_key_and_update_cache=MagicMock(
                return_value=(None, key_scale),
            ),
            update_scale_cache=MagicMock(),
        )

        with patch(
            "vllm_ascend.models.deepseek_v4.indexer.rotate_activation",
            return_value=rotated_key,
        ) as rotate:
            indexer.update_cache(hidden_states, kv_cache, metadata)

        indexer.ops.unpack_dsa_indexer_kv_cache.assert_called_once_with(kv_cache)
        compressor.assert_called_once_with(
            hidden_states=hidden_states,
            state_cache=state_cache,
            metadata=metadata.compressor,
        )
        rotate.assert_called_once_with(key, metadata.compressor.cache.hadamard)
        indexer.ops.quantize_key_and_update_cache.assert_called_once_with(
            rotated_key,
            key_cache,
            full_cache,
            slot_mapping,
        )
        indexer.ops.update_scale_cache.assert_called_once_with(
            key_scale,
            scale_cache,
            slot_mapping,
        )


class TestIndexerForward:
    @pytest.mark.parametrize("use_multistream", [False, True])
    def test_routes_serial_and_multistream_paths(self, use_multistream: bool):
        indexer = _make_indexer(None)
        indexer.skip_topk = False
        indexer.use_index_cache = False
        execution_order: list[str] = []
        topk_indices = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])
        indexer_q = torch.ones((2, 2, 4))

        def select_serial(*args: Any, **kwargs: Any) -> torch.Tensor:
            assert kwargs == {"write_cache": True}
            execution_order.append("select")
            return topk_indices

        def compute_multistream(*args: Any) -> torch.Tensor:
            execution_order.append("compute")
            return indexer_q

        serial_select = MagicMock(side_effect=select_serial)
        multistream_compute = MagicMock(side_effect=compute_multistream)

        def select_multistream(*args: Any) -> torch.Tensor:
            execution_order.append("select_start")
            args[-1]()
            execution_order.append("select_end")
            return topk_indices

        multistream_select = MagicMock(side_effect=select_multistream)
        indexer._select_topk_serial = serial_select
        indexer._cv_compute_query_and_update_cache_multistream = multistream_compute
        indexer._select_topk_multistream = multistream_select
        metadata, cos, sin = _make_forward_metadata()
        compressed_kv = torch.ones((1, 1, 4))
        slot_mapping = torch.zeros((1, 2), dtype=torch.int32)

        def compute_attention_compressed_kv() -> tuple[torch.Tensor, torch.Tensor]:
            execution_order.append("compute_attention_compressed_kv")
            return compressed_kv, slot_mapping

        def scatter_attention_compressed_kv(
            actual_compressed_kv: torch.Tensor,
            actual_slot_mapping: torch.Tensor,
        ) -> None:
            assert actual_compressed_kv is compressed_kv
            assert actual_slot_mapping is slot_mapping
            execution_order.append("scatter_attention_compressed_kv")

        actual = indexer(
            layer_name="layer",
            hidden_states=torch.ones((2, 4)),
            qr=torch.ones((2, 4)),
            kv_cache=(torch.empty(0),),
            metadata=metadata,
            overlap_plan=IndexerOverlapPlan(
                compute_attention_compressed_kv=compute_attention_compressed_kv,
                scatter_attention_compressed_kv=scatter_attention_compressed_kv,
                aux_stream=object() if use_multistream else None,
            ),
        )

        assert actual is topk_indices
        indexer_call = multistream_compute.call_args if use_multistream else serial_select.call_args
        assert indexer_call.args[3] is metadata
        assert torch.equal(indexer_call.args[4], cos)
        assert torch.equal(indexer_call.args[5], sin)
        if use_multistream:
            assert execution_order == [
                "compute",
                "compute_attention_compressed_kv",
                "select_start",
                "scatter_attention_compressed_kv",
                "select_end",
            ]
            serial_select.assert_not_called()
        else:
            assert execution_order == [
                "select",
                "compute_attention_compressed_kv",
                "scatter_attention_compressed_kv",
            ]
            multistream_compute.assert_not_called()
            multistream_select.assert_not_called()

    def test_skip_topk_reuses_cache_and_updates_attention_cache(self):
        topk_indices_buffer = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            dtype=torch.int32,
        )
        indexer = _make_indexer(topk_indices_buffer)
        indexer.skip_topk = True
        indexer.use_index_cache = True
        indexer._select_topk_serial = MagicMock()
        indexer._cv_compute_query_and_update_cache_multistream = MagicMock()
        indexer._select_topk_multistream = MagicMock()
        metadata, _, _ = _make_forward_metadata()
        compressed_kv = torch.ones((1, 1, 4))
        slot_mapping = torch.zeros((1, 2), dtype=torch.int32)
        compute = MagicMock(return_value=(compressed_kv, slot_mapping))
        scatter = MagicMock()

        actual = indexer(
            layer_name="layer",
            hidden_states=torch.ones((2, 4)),
            qr=torch.ones((2, 4)),
            kv_cache=(torch.empty(0),),
            metadata=metadata,
            overlap_plan=IndexerOverlapPlan(
                compute_attention_compressed_kv=compute,
                scatter_attention_compressed_kv=scatter,
                aux_stream=object(),
            ),
        )

        assert torch.equal(actual, topk_indices_buffer[:2].unsqueeze(1))
        indexer._select_topk_serial.assert_not_called()
        indexer._cv_compute_query_and_update_cache_multistream.assert_not_called()
        indexer._select_topk_multistream.assert_not_called()
        compute.assert_called_once_with()
        assert scatter.call_count == 1
        assert scatter.call_args.args[0] is compressed_kv
        assert scatter.call_args.args[1] is slot_mapping

    def test_serial_prepared_cache_selects_topk_without_cache_writes(self):
        indexer = _make_indexer(None)
        indexer.skip_topk = False
        indexer.use_index_cache = False
        indexer.n_heads = 1
        indexer.head_dim = 4
        indexer.rope_head_dim = 2
        indexer.softmax_scale = 1.0
        indexer.wq_b = MagicMock(side_effect=lambda value: value)
        indexer.weights_proj = MagicMock(return_value=torch.ones((2, 1)))
        indexer.compressor = MagicMock()
        indexer.ops = MagicMock()
        key_cache = object()
        scale_cache = object()
        indexer.ops.unpack_dsa_indexer_kv_cache.return_value = (
            object(),
            key_cache,
            scale_cache,
            object(),
        )
        quantized_query = torch.ones((2, 1, 4), dtype=torch.int8)
        query_scale = torch.ones((2, 1))
        topk_indices = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])
        indexer.ops.quantize_query.return_value = (quantized_query, query_scale)
        indexer.ops.select_topk.return_value = topk_indices
        metadata, _, _ = _make_forward_metadata()
        compute = MagicMock()
        scatter = MagicMock()

        with (
            patch(
                "vllm_ascend.models.deepseek_v4.indexer._is_w8a8_dynamic",
                return_value=False,
            ),
            patch.object(
                torch.ops._C_ascend,
                "inplace_partial_rotary_mul",
                create=True,
            ),
        ):
            actual = indexer(
                layer_name="layer",
                hidden_states=torch.ones((2, 4)),
                qr=torch.ones((2, 4)),
                kv_cache=(torch.empty(0),),
                metadata=metadata,
                overlap_plan=IndexerOverlapPlan(
                    compute_attention_compressed_kv=compute,
                    scatter_attention_compressed_kv=scatter,
                    aux_stream=None,
                ),
                write_cache=False,
            )

        assert actual is topk_indices
        indexer.compressor.assert_not_called()
        indexer.ops.quantize_update_cache_and_select_topk.assert_not_called()
        indexer.ops.quantize_query.assert_called_once()
        select_args = indexer.ops.select_topk.call_args.args
        assert select_args[0] is quantized_query
        assert torch.equal(select_args[1], torch.ones((2, 1)))
        assert select_args[2] is query_scale
        assert select_args[3] is key_cache
        assert select_args[4] is scale_cache
        assert select_args[5] is metadata.compressor.cache.req_metadata
        compute.assert_not_called()
        scatter.assert_not_called()


class TestIndexerOps:
    def test_quantize_scatter_then_select_topk(self):
        indexer_ops = AscendIndexerOps(index_topk=3)
        key_cache = torch.empty((1, 1, 1, 4), dtype=torch.int8)
        scale_cache = torch.empty((1, 1, 1, 1), dtype=torch.float16)
        full_cache = torch.empty((1, 1, 1, 4), dtype=torch.uint8)
        query = torch.ones((2, 2, 4))
        quantized_query = torch.ones((2, 2, 4), dtype=torch.int8)
        query_scale = torch.ones((2, 2), dtype=torch.float16)
        key = torch.ones((1, 1, 4))
        weights = torch.ones((2, 2))
        slot_mapping = torch.zeros((1, 2), dtype=torch.int32)
        topk_indices = torch.tensor([[[1, 2, 3]]], dtype=torch.int32)
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            qli_metadata=torch.empty(0, dtype=torch.int32),
        )

        with (
            patch.object(
                DeviceOperator,
                "indexer_quant_scatter",
                return_value=(quantized_query, query_scale, key, None),
            ) as quant_scatter,
            patch.object(
                DeviceOperator,
                "prepare_dsa_indexer_weights",
                side_effect=lambda value: value,
            ),
            patch.object(
                DeviceOperator,
                "prepare_dsa_indexer_query_scale",
                side_effect=lambda value: value,
            ),
            patch.object(
                DeviceOperator,
                "prepare_dsa_indexer_key_scale",
                side_effect=lambda value: value,
            ),
            patch.object(
                torch.ops._C_ascend,
                "npu_vllm_quant_lightning_indexer",
                create=True,
                return_value=(topk_indices, None),
            ) as qli,
        ):
            actual = indexer_ops.quantize_update_cache_and_select_topk(
                query,
                key,
                weights,
                key_cache,
                scale_cache,
                full_cache,
                slot_mapping,
                metadata,
            )

        assert actual is topk_indices
        quant_scatter.assert_called_once_with(
            query,
            key,
            key_cache,
            scale_cache,
            full_cache,
            slot_mapping,
        )
        qli_kwargs = qli.call_args.kwargs
        assert qli_kwargs["query"] is quantized_query
        assert qli_kwargs["key"] is key_cache
        assert qli_kwargs["key_dequant_scale"] is scale_cache
        assert torch.equal(
            qli_kwargs["actual_seq_lengths_query"],
            metadata.query_start_loc[1:],
        )
        assert qli_kwargs["actual_seq_lengths_key"] is metadata.seq_lens
        assert qli_kwargs["block_table"] is metadata.block_table
        assert qli_kwargs["metadata"] is metadata.qli_metadata
