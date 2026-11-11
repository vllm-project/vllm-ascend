from unittest.mock import MagicMock, patch

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import MooncakeAgentMetadata, MooncakeConnector


def test_mooncake_handshake_metadata_only_reaches_mooncake_child():
    wrapper = AscendMultiConnector.__new__(AscendMultiConnector)
    mooncake = MooncakeConnector.__new__(MooncakeConnector)
    mooncake.set_xfer_handshake_metadata_pp_aware = MagicMock()  # type: ignore[method-assign]
    other = MagicMock()
    wrapper._connectors = [other, mooncake]
    metadata = {
        (0, 1, 2): MooncakeAgentMetadata(
            engine_id="prefill",
            te_rpc_port=9000,
            kv_group2layeridx={},
            block_size=16,
            kv_caches_base_addr=[],
            block_size_scale=[],
            num_blocks=0,
            block_lens=[],
            block_strides=[],
            handshake_port=41002,
        )
    }

    wrapper.set_xfer_handshake_metadata_pp_aware(metadata)

    mooncake.set_xfer_handshake_metadata_pp_aware.assert_called_once_with(metadata)  # type: ignore[attr-defined]
    other.set_xfer_handshake_metadata_pp_aware.assert_not_called()


def test_other_handshake_metadata_only_reaches_other_child():
    wrapper = AscendMultiConnector.__new__(AscendMultiConnector)
    mooncake = MooncakeConnector.__new__(MooncakeConnector)
    mooncake.set_xfer_handshake_metadata_pp_aware = MagicMock()  # type: ignore[method-assign]
    other = MagicMock()
    wrapper._connectors = [mooncake, other]
    metadata = {(0, 0): MagicMock()}

    wrapper.set_xfer_handshake_metadata_pp_aware(metadata)

    other.set_xfer_handshake_metadata_pp_aware.assert_called_once_with(metadata)
    mooncake.set_xfer_handshake_metadata_pp_aware.assert_not_called()  # type: ignore[attr-defined]


def test_mixed_handshake_metadata_is_split_by_child_type():
    wrapper = AscendMultiConnector.__new__(AscendMultiConnector)
    mooncake = MooncakeConnector.__new__(MooncakeConnector)
    mooncake.set_xfer_handshake_metadata_pp_aware = MagicMock()  # type: ignore[method-assign]
    other = MagicMock()
    wrapper._connectors = [other, mooncake]
    mooncake_value = MooncakeAgentMetadata(
        engine_id="prefill",
        te_rpc_port=9000,
        kv_group2layeridx={},
        block_size=16,
        kv_caches_base_addr=[],
        block_size_scale=[],
        num_blocks=0,
        block_lens=[],
        block_strides=[],
        handshake_port=41002,
    )
    other_value = MagicMock()
    wrapper.set_xfer_handshake_metadata_pp_aware({(0, 0): mooncake_value, (0, 1): other_value})

    mooncake.set_xfer_handshake_metadata_pp_aware.assert_called_once_with({(0, 0): mooncake_value})  # type: ignore[attr-defined]
    other.set_xfer_handshake_metadata_pp_aware.assert_called_once_with({(0, 1): other_value})


def test_empty_handshake_metadata_keeps_upstream_routing():
    wrapper = AscendMultiConnector.__new__(AscendMultiConnector)
    wrapper._connectors = [MagicMock()]

    with patch.object(MultiConnector, "set_xfer_handshake_metadata_pp_aware") as upstream:
        wrapper.set_xfer_handshake_metadata_pp_aware({})

    upstream.assert_called_once_with({})
