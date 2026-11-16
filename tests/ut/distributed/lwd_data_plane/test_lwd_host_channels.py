# SPDX-License-Identifier: Apache-2.0
"""Host channel bootstrap: one listener per cloud DP, shared by its edges.

The listener binds in its constructor, so a second edge connected to the
same cloud DP must reuse that DP's listener instead of binding data_port
again. Edges and sidecar engines build no engine-side listener at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.config.lwd import LwdConfig, LwdTopology

from vllm_ascend.distributed.lwd_data_plane.host import lwd_host_channels
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

CLOUD_ADDR = "10.1.0.1"
CLOUD_DATA_PORT = 7000


class _Listener:
    """Stand-in for LwdUcxListener: records construction and start()."""

    created: list[tuple[str, int]] = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.started = False
        _Listener.created.append((host, port))

    def start(self) -> None:
        self.started = True


class _Io:
    """Stand-in for LwdNpuUcxChannelIo: keeps the config it was built with."""

    def __init__(self, channel: LwdChannelType, stream: Any, config: Any) -> None:
        self.channel = channel
        self.config = config

    def wait_ready(self, timeout: float) -> bool:
        return True


def _instance(identity: int, prefix: str, *, cloud: bool) -> dict[str, Any]:
    dp: dict[str, Any] = {"dp_idx": 0, "addr": f"{prefix}.{identity + 1}"}
    if cloud:
        dp["ctrl_port"] = 5550 + identity
        dp["data_port"] = CLOUD_DATA_PORT + identity
    # The scene validator needs coordination on every instance; an edge also
    # needs an HTTP endpoint and a key file.
    dp_coordination: dict[str, Any] = {"enabled": True}
    if not cloud:
        dp_coordination["control_url"] = f"http://10.9.0.{identity + 1}:8100"
        dp_coordination["tenant_key_file"] = f"/etc/lwd/tenant-{identity}.key"
    return {"id": identity, "dp": [dp], "coordination": dp_coordination}


def _host_config(role: str, *, deployment: str = "in_process") -> LwdConfig:
    """cloud_share (2 edges, 1 cloud) and lwd_cluster (1 edge, 2 clouds)."""
    edges = 2 if role == "cloud" else 1
    clouds = 1 if role == "cloud" else 2
    topology = LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "cloud_share" if role == "cloud" else "lwd_cluster",
                "edges_num": edges,
                "clouds_num": clouds,
                "data_plane_comm": "host",
                "data_plane_host": {
                    "down_buffer_size": 32,
                    "deployment": deployment,
                },
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": [_instance(identity, "10.0.0", cloud=False) for identity in range(edges)],
            "clouds": [_instance(identity, "10.1.0", cloud=True) for identity in range(clouds)],
            "instance_links": [{"edge": edge, "cloud": cloud} for edge in range(edges) for cloud in range(clouds)],
        }
    )
    return LwdConfig.from_topology("host_topology.yaml", role, 0, topology)


def _single_instance_config(role: str, *, dp_count: int = 2) -> LwdConfig:
    """single_instance host layout: one edge, one cloud, several DPs."""

    def instance(identity: int, prefix: str, *, cloud: bool) -> dict[str, Any]:
        dps: list[dict[str, Any]] = []
        for dp_idx in range(dp_count):
            dp: dict[str, Any] = {
                "dp_idx": dp_idx,
                "addr": f"{prefix}.{identity + 1}",
            }
            if cloud:
                dp["ctrl_port"] = 5550 + dp_idx
                dp["data_port"] = CLOUD_DATA_PORT + dp_idx
            dps.append(dp)
        return {"id": identity, "dp": dps}

    topology = LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "single_instance",
                "edges_num": 1,
                "clouds_num": 1,
                "data_plane_comm": "host",
                "data_plane_host": {"down_buffer_size": 32},
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": [instance(0, "10.0.0", cloud=False)],
            "clouds": [instance(0, "10.1.0", cloud=True)],
            "instance_links": [{"edge": 0, "cloud": 0}],
        }
    )
    return LwdConfig.from_topology("host_topology.yaml", role, 0, topology)


def _edge_share_config() -> LwdConfig:
    """edge_share: one physical edge engine serves two logical edges, each
    connected to its own cloud DP over a one-to-one instance link."""
    topology = LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "edge_share",
                "edges_num": 2,
                "clouds_num": 2,
                "data_plane_comm": "host",
                "data_plane_host": {"down_buffer_size": 32},
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": [_instance(identity, "10.0.0", cloud=False) for identity in range(2)],
            "clouds": [_instance(identity, "10.1.0", cloud=True) for identity in range(2)],
            "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 1}],
        }
    )
    return LwdConfig.from_topology("host_topology.yaml", "edge", 0, topology)


@pytest.fixture(autouse=True)
def _host_bootstrap(monkeypatch) -> None:
    """Isolate the module registries and stub every device-side dependency."""
    _Listener.created = []
    monkeypatch.setattr(lwd_host_channels, "_UCX_IOS", {})
    monkeypatch.setattr(lwd_host_channels, "_UCX_LISTENERS", {})
    monkeypatch.setattr(lwd_host_channels, "_UCX_CHANNELS_INITIALIZED", False)
    monkeypatch.setattr(lwd_host_channels, "LwdUcxListener", _Listener)
    monkeypatch.setattr(lwd_host_channels, "LwdNpuUcxChannelIo", _Io)
    monkeypatch.setattr(
        lwd_host_channels.torch,
        "npu",
        SimpleNamespace(Stream=lambda **_: object()),
        raising=False,
    )
    monkeypatch.setattr(lwd_host_channels, "_resolve_up_block_mb", lambda: 8)
    monkeypatch.setattr(
        lwd_host_channels,
        "_qos_data_plane",
        lambda: SimpleNamespace(traffic_class=None, sl=None),
    )


def _init(monkeypatch, config: LwdConfig, *, dp_identity: tuple[int, int] = (0, 0)) -> None:
    monkeypatch.setattr(
        lwd_host_channels,
        "get_current_vllm_config",
        lambda: SimpleNamespace(lwd_config=config),
    )
    monkeypatch.setattr(lwd_host_channels, "_my_dp_identity", lambda: dp_identity)
    lwd_host_channels.lwd_init_host_channels()


def test_two_edges_of_one_cloud_dp_share_one_listener(monkeypatch) -> None:
    """Regression: the second edge bound data_port again and broke 2E1C."""
    _init(monkeypatch, _host_config("cloud"))

    assert _Listener.created == [(CLOUD_ADDR, CLOUD_DATA_PORT)]
    listener = lwd_host_channels._UCX_LISTENERS[(0, 0)]
    assert listener.started
    ios = lwd_host_channels._UCX_IOS
    assert len(ios) == 4
    for edge_id in (0, 1):
        up = ios[(edge_id, 0, 0, LwdChannelType.UP, "recv")]
        down = ios[(edge_id, 0, 0, LwdChannelType.DOWN, "send")]
        assert up.config.listener is listener
        assert down.config.listener is listener


def test_edge_builds_no_listener(monkeypatch) -> None:
    """Edges connect out to every cloud DP and never listen."""
    _init(monkeypatch, _host_config("edge"))

    assert _Listener.created == []
    assert lwd_host_channels._UCX_LISTENERS == {}
    ios = lwd_host_channels._UCX_IOS
    assert len(ios) == 4
    assert all(config.listener is None for config in (io.config for io in ios.values()))


def test_edge_share_shared_engine_owns_every_logical_edge(monkeypatch) -> None:
    """The one physical edge engine serves both frontends, so it must hold the
    IO of every logical edge; a per-instance view would drop the non-primary
    edge's links and fail its first UP submit with "no host send IO"."""
    _init(monkeypatch, _edge_share_config())

    assert _Listener.created == []
    assert lwd_host_channels._UCX_LISTENERS == {}
    ios = lwd_host_channels._UCX_IOS
    assert set(ios) == {
        (0, 0, 0, LwdChannelType.UP, "send"),
        (0, 0, 0, LwdChannelType.DOWN, "recv"),
        (1, 1, 0, LwdChannelType.UP, "send"),
        (1, 1, 0, LwdChannelType.DOWN, "recv"),
    }
    # Each logical edge connects out to its OWN cloud, not only the primary's.
    assert ios[(0, 0, 0, LwdChannelType.UP, "send")].config.host == "10.1.0.1"
    assert ios[(1, 1, 0, LwdChannelType.UP, "send")].config.host == "10.1.0.2"


def test_edge_owns_every_cloud_dp_link(monkeypatch) -> None:
    """One edge engine serves both cloud DPs.

    Dropping the non-primary DP's connection (dp_idx != the edge's own, always
    0) leaves that DP without IO, and its first UP submit fails with
    "no host send IO".
    """
    _init(monkeypatch, _single_instance_config("edge"))

    assert _Listener.created == []
    assert set(lwd_host_channels._UCX_IOS) == {
        (0, 0, 0, LwdChannelType.UP, "send"),
        (0, 0, 0, LwdChannelType.DOWN, "recv"),
        (0, 0, 1, LwdChannelType.UP, "send"),
        (0, 0, 1, LwdChannelType.DOWN, "recv"),
    }


def test_cloud_dp_owns_only_its_own_link(monkeypatch) -> None:
    """Each cloud DP engine binds its own listener and holds its own link."""
    _init(monkeypatch, _single_instance_config("cloud"), dp_identity=(1, 0))

    assert _Listener.created == [(CLOUD_ADDR, CLOUD_DATA_PORT + 1)]
    assert set(lwd_host_channels._UCX_IOS) == {
        (0, 0, 1, LwdChannelType.UP, "recv"),
        (0, 0, 1, LwdChannelType.DOWN, "send"),
    }


def test_sidecar_cloud_leaves_the_listener_to_the_daemon(monkeypatch) -> None:
    """The transport daemon owns the listener; the engine only attaches."""
    _init(monkeypatch, _host_config("cloud", deployment="sidecar"))

    assert _Listener.created == []
    assert lwd_host_channels._UCX_LISTENERS == {}
    ios = lwd_host_channels._UCX_IOS
    assert {io.config.transport_mode for io in ios.values()} == {"daemon"}


def test_dp_identity_survives_the_independent_dp_reset(monkeypatch) -> None:
    """A non-MoE DP engine resets `data_parallel_rank` to 0.

    The host endpoint must key on `data_parallel_index` (the launched DP rank);
    otherwise every cloud DP claims dp0 and binds the same listener port.
    """
    monkeypatch.setattr(
        lwd_host_channels,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=1, data_parallel_rank=0)),
    )
    monkeypatch.setattr(lwd_host_channels, "get_tensor_model_parallel_rank", lambda: 0)

    assert lwd_host_channels._my_dp_identity() == (1, 0)


def test_dp_identity_falls_back_to_the_rank_field(monkeypatch) -> None:
    """Configs without the index field keep their previous meaning."""
    monkeypatch.setattr(
        lwd_host_channels,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=2)),
    )
    monkeypatch.setattr(lwd_host_channels, "get_tensor_model_parallel_rank", lambda: 3)

    assert lwd_host_channels._my_dp_identity() == (2, 3)
