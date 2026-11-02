# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-specific V4.1 PCP metadata and shared-buffer regressions."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.ut.attention.test_dsa_v41_cp import LAYER, PREFIXES, _batch, _build_planes
from tests.ut.attention.test_dsa_v41_cp import builders as pcp_builders
from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.context_parallel import dsa_cp, dsa_v41_cp
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

builders = pcp_builders


class TestSharedPCPMetadata:
    @pytest.mark.parametrize("kind", ["swa", "long_kv", "index_k", "compressor_state"])
    def test_request_metadata_accepts_optional_common_fields(self, builders, kind):
        owner = builders(specialized=True)[kind]
        common = _batch(2, 0, "prefill").local
        common = SimpleNamespace(
            **{
                name: value
                for name, value in vars(common).items()
                if name not in {"attn_state", "is_prefilling", "causal"}
            }
        )
        metadata = owner._global_builder.build(0, common)
        assert metadata.attn_state is None
        assert metadata.is_prefilling is None
        assert metadata.causal is True

    @pytest.mark.parametrize("scenario", ["prefill", "decode", "dummy", "empty"])
    def test_roles_assemble_metadata_without_generic_dispatch(self, builders, monkeypatch, scenario):
        owners = builders(specialized=True)
        forbidden = Mock(side_effect=AssertionError("Unrelated metadata consumer was invoked"))
        monkeypatch.setattr(dsa_v41.AscendDSAV41MetadataBuilder, "build_req_metadata", forbidden)
        for kind, builder in owners.items():
            for consumer in (builder, builder._global_builder):
                if kind == "compressor_state":
                    monkeypatch.setattr(consumer, "_prepare_query_inputs", forbidden)
                    monkeypatch.setattr(consumer, "_build_cache_slots", forbidden)
                elif kind == "index_k":
                    monkeypatch.setattr(consumer, "_build_attention_query", forbidden)
                    monkeypatch.setattr(consumer, "_build_smla_metadata", forbidden)
                if consumer is builder._global_builder:
                    monkeypatch.setattr(consumer, "_prepare_query_inputs", forbidden)
        _build_planes(owners, _batch(2, 0, scenario))
        forbidden.assert_not_called()

    def test_state_cache_does_not_require_query_configuration(self, builders):
        owners = builders(specialized=True)
        state = owners["compressor_state"]
        config = state.vllm_config.model_config.hf_text_config
        del config.num_attention_heads
        del config.index_topk
        batch = _batch(2, 0, "prefill")
        metadata = state.build(0, batch.local, pcp_context=batch.context, pcp_cache_group_idx=0)
        assert metadata.c2_ring_metadata is None
        assert metadata.global_metadata.c2_ring_metadata is not None
        assert metadata.global_metadata.smla_metadata is None
        assert metadata.global_metadata.qli_metadata is None

    @pytest.mark.parametrize("scenario", ["prefill", "decode"])
    def test_folded_index_only_builds_resource_metadata(self, builders, monkeypatch, scenario):
        owner = builders(specialized=True)["index_k"]
        spec = AscendMLAAttentionSpec(
            block_size=8, num_kv_heads=1, head_size=544, dtype=torch.uint8, tokens_per_state=8
        )
        builder = dsa_v41_cp.AscendDSAV41PCPFoldedIndexMetadataBuilder(
            spec, owner.layer_names, owner.vllm_config, torch.device("cpu")
        )
        forbidden = Mock(side_effect=AssertionError("Folded resource must not prepare queries or writes"))
        for consumer in (builder, builder._global_builder):
            monkeypatch.setattr(consumer, "_prepare_query_inputs", forbidden)
            monkeypatch.setattr(consumer, "_build_cache_slots", forbidden)
            monkeypatch.setattr(consumer, "_prepare_cache_lengths", forbidden)
        batch = _batch(2, 0, scenario)
        metadata = builder.build(0, batch.local, pcp_context=batch.context, pcp_cache_group_idx=0)
        for plane in (metadata, metadata.global_metadata):
            if plane is not None:
                assert plane.cache_kind == "index_k_folded"
                assert plane.slot_mapping.numel() == 0
                assert plane.smla_metadata is None
                assert plane.qli_metadata is None
        forbidden.assert_not_called()

    @pytest.mark.parametrize("scenario", ["prefill", "mixed", "decode", "dummy", "empty"])
    @pytest.mark.parametrize("decode_sharded", [False, True])
    @pytest.mark.parametrize("rank", [0, 1])
    @pytest.mark.parametrize("ratio", [1, 2])
    def test_shared_matches_consumed_fields(self, builders, scenario, decode_sharded, rank, ratio):
        reference = _build_planes(
            builders(rank=rank, decode_sharded=decode_sharded, ratio=ratio),
            _batch(2, rank, scenario, decode_sharded=decode_sharded),
        )
        owners = builders(rank=rank, decode_sharded=decode_sharded, specialized=True, ratio=ratio)
        actual = _build_planes(owners, _batch(2, rank, scenario, decode_sharded=decode_sharded))
        for kind, prefix in PREFIXES.items():
            expected, result = reference[prefix], actual[prefix]
            for name in ("block_table", "query_start_loc", "seq_lens", "positions"):
                torch.testing.assert_close(getattr(result, name), getattr(expected, name))
            if kind != "compressor_state":
                torch.testing.assert_close(result.cache_seq_lens, expected.cache_seq_lens)
                assert result.max_cache_seq_len == expected.max_cache_seq_len
            expected_cache = expected.global_metadata or expected
            result_cache = result.global_metadata or result
            for name in (
                "block_table",
                "slot_mapping",
                "flat_slot_mapping",
                "c2_ring_metadata",
                "c2_complete_mask",
                "c2_source_positions",
                "c2_source_cos",
                "c2_source_sin",
            ):
                torch.testing.assert_close(getattr(result_cache, name), getattr(expected_cache, name))
            if result.global_metadata is not None:
                torch.testing.assert_close(result.hidden_restore_idx, expected.hidden_restore_idx)
                assert result_cache.smla_metadata is None
                assert result_cache.qli_metadata is None
                if kind != "compressor_state" and result.num_actual_tokens:
                    assert result.slot_mapping.numel() == 0
                    assert result.flat_slot_mapping is None
                assert result.c2_ring_metadata is None
            if kind == "swa" and result.num_actual_tokens:
                torch.testing.assert_close(result.cos[LAYER], expected.cos[LAYER])
                torch.testing.assert_close(result_cache.cos[LAYER], expected_cache.cos[LAYER])

    def test_shared_context_and_rope_have_stable_owners(self, builders, monkeypatch):
        owners = builders(specialized=True)
        original = owners["swa"]
        duplicate = type(original)(
            original.kv_cache_spec, original.layer_names, original.vllm_config, torch.device("cpu")
        )
        duplicate.prepare_source_rope()
        owners["second_swa"] = duplicate
        dsa_v41_cp.AscendDSAV41PCPMetadataBuilder.prepare_shared_buffers(owners.values())
        context_owner = original._pcp_context_owner
        prepare = Mock(wraps=context_owner._prepare_graph_pcp_context)
        monkeypatch.setattr(context_owner, "_prepare_graph_pcp_context", prepare)
        gather = Mock(wraps=torch.gather)
        monkeypatch.setattr(torch, "gather", gather)
        addresses = []
        for ordered in (list(owners.values()), list(reversed(list(owners.values())))):
            batch = _batch(2, 0, "prefill")
            shared: dict[tuple[str, int, int | None], Any] = {}
            results = [
                builder.build(
                    0,
                    batch.local,
                    pcp_context=batch.context,
                    pcp_cache_group_idx=0,
                    num_actual_reqs=batch.local.num_reqs,
                    common_v41_metadata={},
                    common_v41_batch_metadata=shared,
                )
                for builder in ordered
            ]
            addresses.append(results[0].hidden_restore_idx.data_ptr())
            assert len({result.hidden_restore_idx.data_ptr() for result in results}) == 1
            swa_results = [result for result in results if result.cache_kind == "swa"]
            assert swa_results[0].cos[LAYER].data_ptr() == swa_results[1].cos[LAYER].data_ptr()
            assert swa_results[0].cos[LAYER].data_ptr() != swa_results[0].global_metadata.cos[LAYER].data_ptr()
        assert addresses[0] == addresses[1]
        assert prepare.call_count == 2  # Once per batch, independent of group count.
        assert gather.call_count == 8  # Two tables (cos/sin), two views, two batches.

    @pytest.mark.parametrize("vision_layers", [0, 24])
    def test_build_always_specializes_without_opt_in(self, builders, vision_layers):
        owners = builders(specialized=True)
        owners["swa"].vllm_config.model_config.hf_text_config.vision_n_layers = vision_layers
        batch = _batch(2, 0, "prefill")
        result = _build_planes(owners, batch)
        assert not hasattr(owners["swa"], "build_shared")
        for kind in ("swa", "long_kv", "index_k"):
            metadata = result[PREFIXES[kind]]
            assert metadata.slot_mapping.numel() == 0
            assert metadata.global_metadata.slot_mapping.numel() > 0
        assert result[PREFIXES["compressor_state"]].c2_ring_metadata is None
        assert result[PREFIXES["compressor_state"]].global_metadata.c2_ring_metadata is not None

    def test_direct_build_without_batch_cache(self, builders):
        builder = builders(specialized=True)["swa"]
        batch = _batch(2, 0, "prefill")
        metadata = builder.build(0, batch.local, pcp_context=batch.context, pcp_cache_group_idx=0)
        assert metadata.slot_mapping.numel() == 0
        assert metadata.global_metadata.slot_mapping.numel() > 0

    @pytest.mark.parametrize("ratio", [1, 2])
    def test_query_operators_only_run_on_local_consumers(self, builders, monkeypatch, ratio):
        owners = builders(specialized=True, ratio=ratio)
        config = owners["swa"].vllm_config.model_config.hf_text_config
        config.index_n_heads = 2
        config.index_head_dim = 2
        for kind in ("swa", "long_kv", "index_k"):
            for builder in (owners[kind], owners[kind]._global_builder):
                builder._supports_device_ops = True
                builder._uses_a5_packed_cache = False
        output = torch.zeros(dsa_v41.V41_METADATA_BUFFER_SIZE, dtype=torch.int32)
        smla = Mock(return_value=output)
        qli = Mock(return_value=output)
        slots = Mock(wraps=dsa_v41.compressed_slot_mapping)
        monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla_metadata", smla, raising=False)
        monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", qli, raising=False)
        monkeypatch.setattr(dsa_v41, "compressed_slot_mapping", slots)
        result = _build_planes(owners, _batch(2, 0, "prefill"))
        assert smla.call_count == 2  # SWA and long-context attention.
        assert qli.call_count == 1  # Only the indexer.
        assert slots.call_count == (2 if ratio == 2 else 0)  # Global writes only.
        assert result[PREFIXES["swa"]].smla_metadata is not None
        assert result[PREFIXES["long_kv"]].smla_metadata is not None
        assert result[PREFIXES["index_k"]].qli_metadata is not None
        for prefix in PREFIXES.values():
            assert result[prefix].global_metadata.smla_metadata is None
            assert result[prefix].global_metadata.qli_metadata is None

    def test_buffer_owners_are_isolated_between_model_configs(self, builders):
        target = builders(specialized=True)
        draft = builders(specialized=True)
        dsa_v41_cp.AscendDSAV41PCPMetadataBuilder.prepare_shared_buffers([*target.values(), *draft.values()])
        assert target["swa"]._pcp_context_owner is not draft["swa"]._pcp_context_owner
        target_rope = target["swa"]._pcp_rope_buffers[LAYER]
        draft_rope = draft["swa"]._pcp_rope_buffers[LAYER]
        for target_buffer, draft_buffer in zip(target_rope[2:], draft_rope[2:]):
            assert target_buffer.data_ptr() != draft_buffer.data_ptr()

    def test_deferred_c2_task_has_one_global_owner(self, builders):
        owners = builders(specialized=True)
        for builder in owners.values():
            builder.enable_device_metadata()
        metadata = _build_planes(owners, _batch(2, 0, "prefill"))
        tasks = [task for builder in owners.values() for task in builder.take_device_metadata_tasks()]
        assert len(tasks) == 1
        assert tasks[0].stage == dsa_v41.DeviceMetadataStage.COMPRESSOR
        tasks[0].run()
        state = metadata[PREFIXES["compressor_state"]]
        assert state.c2_ring_metadata is None
        assert state.global_metadata.c2_ring_metadata[1].tolist() == [8, 8]
        assert all(not builder.take_device_metadata_tasks() for builder in owners.values())

    def test_shared_batch_signature_is_recorded_and_checked(self, builders):
        builder = builders(specialized=True)["swa"]
        batch = _batch(2, 0, "prefill")
        shared: dict[tuple[str, int, int | None], Any] = {}
        kwargs = dict(
            pcp_context=batch.context,
            pcp_cache_group_idx=0,
            common_v41_batch_metadata=shared,
            num_actual_reqs=batch.local.num_reqs,
        )
        builder.build(0, batch.local, **kwargs)
        cache = shared[("v41_pcp_specialized", id(builder.vllm_config), None)]
        # Explicit checks also execute when this regression is run with python -O.
        if "signature" not in cache:
            raise AssertionError("Batch signature was not recorded")
        try:
            builder.build(0, batch.local, skip_ring_state_update=True, **kwargs)
        except ValueError as exc:
            if "matching batch signatures" not in str(exc):
                raise
        else:
            raise AssertionError("Mismatched batch signature was accepted")

    def test_v41_build_does_not_call_v4_instance_helpers(self, builders, monkeypatch):
        forbidden = Mock(side_effect=AssertionError("V4 instance helper called by V4.1"))
        for name in ("_prepare_graph_pcp_context", "_build_local_common_attn_metadata"):
            monkeypatch.setattr(dsa_cp.AscendDSAPCPMetadataBuilder, name, forbidden)
        owners = builders(specialized=True)
        for scenario in ("prefill", "decode", "dummy", "empty"):
            _build_planes(owners, _batch(2, 0, scenario))
        forbidden.assert_not_called()
