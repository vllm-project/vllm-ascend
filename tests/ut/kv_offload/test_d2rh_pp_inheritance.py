# SPDX-License-Identifier: Apache-2.0
"""D2RH owns its extensions without changing the generic V1 connector."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector as v1
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh


@pytest.mark.parametrize("module", [v1, d2rh])
@pytest.mark.parametrize("role", [v1.KVConnectorRole.SCHEDULER, v1.KVConnectorRole.WORKER])
def test_facade_inherits_dispatch_and_selects_correct_implementation(module, role):
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="engine"))
    cache = object()
    with (
        patch.object(module, "MooncakeConnectorScheduler") as scheduler,
        patch.object(module, "MooncakeConnectorWorker") as worker,
    ):
        connector = module.MooncakeConnector(config, role, cache)
        if role == v1.KVConnectorRole.SCHEDULER:
            scheduler.assert_called_once_with(config, "engine", cache)
            worker.assert_not_called()
            connector.get_num_new_matched_tokens("request", 128)
            scheduler.return_value.get_num_new_matched_tokens.assert_called_once_with("request", 128)
        else:
            worker.assert_called_once_with(config, "engine", cache)
            scheduler.assert_not_called()
            connector.register_kv_caches("caches")
            worker.return_value.register_kv_caches.assert_called_once_with("caches")
    assert issubclass(d2rh.MooncakeConnector, v1.MooncakeConnector)
    assert d2rh.MooncakeConnector.__init__ is not v1.MooncakeConnector.__init__
    assert d2rh.MooncakeConnectorWorker.register_kv_caches is not v1.MooncakeConnectorWorker.register_kv_caches


def make_worker(worker_cls, role):
    worker = object.__new__(worker_cls)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_mla=True, hf_text_config=SimpleNamespace()),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda name, default: default),
    )
    worker.kv_cache_config = SimpleNamespace(num_blocks=4)
    worker.block_size = 128
    worker.kv_role = role
    worker.use_hybrid = False
    worker._is_hma_required = True
    worker.engine_id = "engine"
    worker.engine = object()
    worker.te_rpc_port = 1234
    worker.handshake_port = 1235
    worker.side_channel_host = "127.0.0.1"
    worker.side_channel_port = 1236
    worker.tp_rank = worker.pp_rank = worker.pcp_rank = 0
    worker.tp_size = worker._prefill_tp_size = 1
    worker._prefill_pp_size = 2
    worker._prefill_pp_layer_partition = None
    worker.remote_local_block_map = {}
    worker.d2rh_thread = None
    # Two groups own different cache components at physical layer 22;
    # retain sparse metadata indices and do not deduplicate storage aliases.
    worker._build_kv_group2layeridx = lambda: {
        0: ({"layer_names": ["layer22.indexer"], "kv_cache_spec_type": "AttentionSpec"}, [22]),
        1: ({"layer_names": ["layer22.swa"], "kv_cache_spec_type": "AttentionSpec"}, [22]),
    }
    return worker


@pytest.mark.parametrize("worker_cls", [v1.MooncakeConnectorWorker, d2rh.MooncakeConnectorWorker])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_registration_preserves_layout_and_d2rh_owns_host_regions(worker_cls, role):
    worker = make_worker(worker_cls, role)
    runtime_module = d2rh if worker_cls is d2rh.MooncakeConnectorWorker else v1
    tensor = torch.empty((8, 2, 3))
    caches = {"layer22.indexer": (tensor, tensor[:, :, :1]), "layer22.swa": tensor}
    real_empty = torch.empty

    def pinned_empty(*args, **kwargs):
        assert kwargs.pop("pin_memory") is True
        return real_empty(*args, **kwargs)

    def ready_thread(*args, **kwargs):
        event = args[7] if role == "kv_producer" else args[11]
        thread = MagicMock()
        thread.start.side_effect = event.set
        return thread

    with (
        patch.object(runtime_module, "enable_sfa_dcp_replicated_indexer", return_value=False),
        patch.object(runtime_module.global_te, "register_buffer") as register,
        patch.object(runtime_module, "KVCacheSendingThread", side_effect=ready_thread) as sender,
        patch.object(runtime_module, "KVCacheRecvingThread", side_effect=ready_thread) as receiver,
        patch.object(d2rh, "D2RHThread") as hop1,
        patch.object(d2rh, "get_d2rh_zmq_port", return_value=38100),
        patch.object(d2rh, "get_scheduler_ready_zmq_port", return_value=38200),
        patch.object(d2rh.torch, "empty", side_effect=pinned_empty) as allocate,
    ):
        worker.register_kv_caches(caches)
        assert len(worker.kv_caches_base_addr) == 23
        assert worker.kv_caches_base_addr[22] == [tensor.data_ptr()] * 3
        assert worker.block_size_scale[22] == [2, 2, 2]
        assert worker.block_len_per_addr[22] == [24, 8, 24]
        assert worker.block_stride_per_addr[22] == [24, 24, 24]
        assert worker.xfer_handshake_metadata.kv_group2layeridx == worker.kv_group2layeridx
        if worker_cls is d2rh.MooncakeConnectorWorker:
            assert worker.kv_group2layeridx[0][0]["layer_cache_indices"] == {"layer22.indexer": [0, 1]}
            assert worker.kv_group2layeridx[1][0]["layer_cache_indices"] == {"layer22.swa": [2]}
        else:
            assert "layer_cache_indices" not in worker.kv_group2layeridx[0][0]
        if worker_cls is d2rh.MooncakeConnectorWorker and role == "kv_consumer":
            assert allocate.call_count == 1
            assert len(worker.cpu_caches_hold) == 1
            assert len(set(worker.cpu_kv_caches_base_addr[22])) == 2
            assert worker.cpu_block_stride_per_addr[22] == [24, 24, 24]
            assert worker.cpu_block_size_scale[22] == [2, 2, 2]
            assert worker._cpu_register_lengths == [d2rh.HUGEPAGE_SIZE_2M * 2]
            assert worker._cpu_register_ptrs[0] % d2rh.HUGEPAGE_SIZE_2M == 0
            assert register.call_args.args[0][-1:] == worker._cpu_register_ptrs
            assert register.call_args.args[1][-1:] == worker._cpu_register_lengths
            assert len(register.call_args.args[0]) == 2  # One HBM storage and one Host arena.
            assert receiver.call_args.kwargs["cpu_kvcache_manager"] is worker.cpu_kvcache_manager
            assert receiver.call_args.kwargs["remote_local_block_map"] is worker.remote_local_block_map
            hop1.return_value.start.assert_called_once()
        else:
            allocate.assert_not_called()
            hop1.assert_not_called()
            assert len(register.call_args.args[0]) == 1  # HBM storage aliases merged by V1.
        assert sender.call_count == (role == "kv_producer")
        assert receiver.call_count == (role == "kv_consumer")


def test_shared_helpers_are_not_copied():
    for name in ["KVCacheTaskTracker", "zmq_ctx", "ensure_zmq_send", "string_to_int64_hash"]:
        assert getattr(d2rh, name) is getattr(v1, name)
