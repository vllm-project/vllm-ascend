"""Shared transfer values do not depend on concrete Worker routes or their plans."""

import ast
from pathlib import Path

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import base
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import transfer as bulk_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import gva, key_range
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import transfer as layer_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer import batch, completion, rows


def test_transfer_definitions_belong_to_their_consumers():
    owners = {
        base: (base.KVPoolStepContext, base.PendingStore),
        bulk_transfer: (
            bulk_transfer.BulkBackendArguments,
            bulk_transfer.select_store_writer_rows,
            bulk_transfer._bulk_load_completions,
            bulk_transfer._store_completions,
        ),
        layer_transfer: (
            layer_transfer.LayerTransferGroup,
            layer_transfer.LayerTransferPlan,
            layer_transfer.LayerStoreResult,
            layer_transfer.prepare_layer_load,
            layer_transfer.prepare_layer_store,
            layer_transfer._layer_load_completions,
        ),
        key_range: (key_range.KeyRangeArguments, key_range.materialize_load_layer_ranges),
        gva: (gva.materialize_layer_gva,),
        batch: (batch.StoreCandidates, batch.StoreGroupCandidates, batch.store_candidate_keys),
        rows: (rows.merge_key_axes, rows.compact_single_axis_rows),
        completion: (completion.TransferEvidence, completion.LoadCompletion, completion.StoreCompletion),
    }
    for module, definitions in owners.items():
        assert all(definition.__module__ == module.__name__ for definition in definitions)


def test_shared_transfer_does_not_import_route_specific_types():
    root = Path(batch.__file__).parent
    route_modules = {"bulk", "layerwise", "mapping", "base"}
    route_types = {"BulkBackendArguments", "LayerTransferPlan", "LayerTransferGroup", "LayerStoreResult"}
    for source in root.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not route_modules.intersection((node.module or "").split(".")), source
                assert not route_types.intersection(alias.name for alias in node.names), source
            elif isinstance(node, ast.Import):
                assert all(not route_modules.intersection(alias.name.split(".")) for alias in node.names), source
