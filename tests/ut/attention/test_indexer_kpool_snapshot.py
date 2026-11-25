import torch

from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolMetadataBuilder, Glm5NextKPoolIndexerBackend
from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import dump_state_dict, restore_state_dict
from vllm_ascend.snapshot.model_runner_lifecycle.tensor_inventory import capture_tensor_inventory


def test_kpool_builder_discards_per_step_metadata_before_recapture():
    builder = AscendIndexerKPoolMetadataBuilder.__new__(AscendIndexerKPoolMetadataBuilder)
    builder._metadata_buffers = {123: (torch.ones(4),)}
    builder._dsa_cp_query_buffers = {123: torch.ones(4)}
    builder._captured_state_tables = {(1, 2, 3): torch.ones(4)}
    builder._state_table_scratch = {1: torch.ones(4)}
    builder._state_table_shapes = {1: (2, 2)}
    builder.reset_runtime_state_after_snapshot_restore()
    builder.reset_runtime_state_after_snapshot_restore()
    assert builder._metadata_buffers == {}
    assert builder._dsa_cp_query_buffers == {}
    assert builder._captured_state_tables == {}
    assert builder._state_table_scratch == {}
    assert builder._state_table_shapes == {}


def test_kpool_indexer_restores_derived_weights_in_place(tmp_path):
    backend = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend.head_dim = 2
    backend.wk_weights_proj = torch.nn.Linear(3, 4, bias=False)
    backend.index_kpool_compress_gate = torch.nn.Parameter(torch.ones(3))
    backend.k_norm = torch.nn.LayerNorm(3)
    backend.process_weights_after_loading()
    capture_tensor_inventory(backend)
    names = ("_key_weight", "_gate_weight", "_head_weight_f32", "_norm_weight_f32", "_norm_bias_f32")
    original = {name: getattr(backend, name).clone() for name in names}
    addresses = {name: getattr(backend, name).data_ptr() for name in names}
    path = str(tmp_path / "kpool.pth")
    dump_state_dict(backend, path)
    for _ in range(2):
        with torch.no_grad():
            for name in names:
                getattr(backend, name).zero_()
        restore_state_dict(backend, path, "kpool")
        for name in names:
            torch.testing.assert_close(getattr(backend, name), original[name])
            assert getattr(backend, name).data_ptr() == addresses[name]


def test_kpool_indexer_clears_topk_buffer_after_restore():
    backend = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)
    backend.topk_indices_buffer = torch.zeros((2, 4), dtype=torch.int32)

    backend.reset_runtime_state_after_snapshot_restore()

    assert torch.all(backend.topk_indices_buffer == -1)
