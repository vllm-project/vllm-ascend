from pathlib import Path

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ExternalDPConfigLoader,
    RankResolver,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import (
    build_proxy_server_cmd,
    wait_ranks_ready,
)


def test_headless_pp_peer_is_not_routed_or_health_polled(tmp_path: Path):
    config_path = tmp_path / "pd.yaml"
    config_path.write_text(
        """
model: /models/glm
num_nodes: 3
npu_per_node: 8
routing:
  type: disaggregated_prefill
  groups: {prefiller: [0, 1], decoder: [2]}
config:
  - node_index: 0
    port_start: 7100
    dp_rpc_port: 12321
    dp_size: 1
    dp_size_local: 1
    dp_rank_start: 0
    tp_size: 8
    dp_address: "${NODE_0_IP}"
  - node_index: 1
    port_start: 7100
    dp_rpc_port: 12321
    dp_size: 1
    dp_size_local: 1
    dp_rank_start: 0
    tp_size: 8
    dp_address: "${NODE_0_IP}"
    headless: true
  - node_index: 2
    port_start: 7200
    dp_rpc_port: 12322
    dp_size: 8
    dp_size_local: 8
    dp_rank_start: 0
    tp_size: 1
    dp_address: "${NODE_2_IP}"
templates:
  - {node_index: 0, envs: {}, server_cmd_template: [--port, "${PORT}"]}
  - {node_index: 1, envs: {}, server_cmd_template: [--headless]}
  - {node_index: 2, envs: {}, server_cmd_template: [--port, "${PORT}"]}
benchmarks: {}
""",
        encoding="utf-8",
    )
    config = ExternalDPConfigLoader.from_yaml(str(config_path), cluster_ips=["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    ranks = RankResolver(config).resolve()

    assert ranks[1].headless is True
    proxy_cmd = build_proxy_server_cmd(config, ranks)
    assert "10.0.0.1" in proxy_cmd
    assert "10.0.0.2" not in proxy_cmd
    wait_ranks_ready([ranks[1]], timeout=0)
