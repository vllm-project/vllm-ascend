# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)

from .helpers import make_full_spec, make_mamba_spec, make_sliding_spec


def make_scheduler_config(
    *,
    is_consumer: bool = True,
    is_producer: bool = False,
    pcp_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_consumer=is_consumer,
            is_kv_producer=is_producer,
            kv_role="kv_consumer" if is_consumer else "kv_producer",
            kv_port=6000,
        ),
        cache_config=SimpleNamespace(block_size=16),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=2,
            data_parallel_size=3,
            data_parallel_rank=1,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(compress_ratios=None)),
    )


def test_base_scheduler_initializes_parallel_layout_and_control_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_scheduler_config()
    group = KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=make_full_spec())
    kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
    ascend_config = object()
    init_ascend_config = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler.init_ascend_config",
        init_ascend_config,
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler.get_ascend_config",
        MagicMock(return_value=ascend_config),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler.get_ip",
        MagicMock(return_value="10.0.0.1"),
    )

    scheduler = MooncakeBaseConnectorScheduler(config, "engine-d", kv_cache_config)  # type: ignore[arg-type]

    assert scheduler.engine_id == "engine-d"
    assert scheduler.block_size == 16
    assert scheduler.num_speculative_tokens == 3
    assert scheduler.ascend_config is ascend_config
    assert scheduler.side_channel_host == "10.0.0.1"
    assert scheduler.max_device_id == 24
    assert scheduler.side_channel_port == 6025
    assert scheduler.group_block_size == [16]
    assert scheduler.group_unique_specs == [[group.kv_cache_spec]]
    assert scheduler.need_truncate is False
    init_ascend_config.assert_called_once_with(config)


@pytest.mark.parametrize(("is_consumer", "is_producer"), [(False, False), (True, True)])
def test_base_scheduler_rejects_invalid_transfer_roles(
    is_consumer: bool,
    is_producer: bool,
) -> None:
    config = make_scheduler_config(is_consumer=is_consumer, is_producer=is_producer)

    with pytest.raises(ValueError, match="exactly one KV transfer role"):
        MooncakeBaseConnectorScheduler(config, "engine", SimpleNamespace())  # type: ignore[arg-type]


def test_base_scheduler_rejects_unsupported_pcp(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_scheduler_config(pcp_size=2)
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler.init_ascend_config",
        MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler.get_ascend_config",
        MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler.get_ip",
        MagicMock(return_value="10.0.0.1"),
    )

    with pytest.raises(AssertionError, match="prefill context parallel size 1"):
        MooncakeBaseConnectorScheduler(config, "engine", SimpleNamespace())  # type: ignore[arg-type]


def test_base_scheduler_expands_unique_uniform_specs_in_layer_order() -> None:
    full = make_full_spec()
    sliding = make_sliding_spec()
    uniform = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer.0": full, "layer.1": full, "layer.2": sliding},
    )
    group = KVCacheGroupSpec(
        layer_names=["layer.1", "layer.0", "layer.2"],
        kv_cache_spec=uniform,
    )

    assert MooncakeBaseConnectorScheduler._get_group_unique_specs(group) == [full, sliding]


def test_base_scheduler_transfer_blocks_handles_empty_and_state_without_speculation() -> None:
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.pcp_size = 1
    scheduler.dcp_size = 1
    scheduler.num_speculative_tokens = 0
    scheduler.group_block_size = [16]
    scheduler.group_unique_specs = [[make_mamba_spec()]]

    assert scheduler._get_transfer_block_ids((), prompt_len=32) == ()
    assert scheduler._get_transfer_block_ids(([10, 11],), prompt_len=32) == ([10, 11],)


def test_base_scheduler_abstract_contract_and_legacy_metadata_delegation() -> None:
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)

    with pytest.raises(NotImplementedError):
        scheduler.on_new_request(MagicMock())
    with pytest.raises(NotImplementedError):
        scheduler.update_connector_output(MagicMock())
    with pytest.raises(NotImplementedError):
        scheduler.get_num_new_matched_tokens(MagicMock(), 0)
    with pytest.raises(NotImplementedError):
        scheduler.update_state_after_alloc(MagicMock(), MagicMock(), 0)
    with pytest.raises(NotImplementedError):
        scheduler.build_connector_meta(MagicMock())
    with pytest.raises(NotImplementedError):
        scheduler.request_finished(MagicMock(), ())

    scheduler.set_xfer_handshake_metadata_from_workers = MagicMock()  # type: ignore[method-assign]
    metadata = {0: MagicMock()}
    scheduler.set_xfer_handshake_metadata(metadata)
    scheduler.set_xfer_handshake_metadata_from_workers.assert_called_once_with(metadata)
