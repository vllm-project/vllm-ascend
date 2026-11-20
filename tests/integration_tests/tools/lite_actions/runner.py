"""Launch the shared Lite Actions collective on the leased nodes."""

import os
import subprocess
import sys

from tests.integration_tests import OverrideDefinitions


def run_dispatch_smoke(test: OverrideDefinitions) -> None:
    """Launch a small HCCL collective without loading model/checkpoint assets."""
    nodes = int(os.environ.get("NNODES", str(test.nnodes)))
    node_ips = [value.strip() for value in os.environ.get("NODE_IPS", "").split(",") if value.strip()]
    if nodes != test.nnodes or (nodes > 1 and len(node_ips) != nodes) or (node_ips and len(node_ips) != nodes):
        raise RuntimeError(f"smoke topology mismatch: nodes={nodes}, ips={node_ips}")
    if os.environ.get("NGPU", str(test.ngpu)) != str(test.ngpu):
        raise RuntimeError(f"smoke expects NGPU={test.ngpu}")
    if "NODE_RANK" in os.environ:
        node_rank = int(os.environ["NODE_RANK"])
    elif nodes == 1:
        node_rank = 0
    else:
        raise RuntimeError("multi-node smoke requires dispatcher-assigned NODE_RANK")
    if not 0 <= node_rank < nodes:
        raise RuntimeError(f"invalid NODE_RANK={node_rank} for {nodes} nodes")

    if os.environ.get("LITE_CI_JOB_ID") and (
        not node_ips
        or any(
            not os.environ.get(name)
            for name in (
                "MASTER_PORT",
                "HCCL_IF_BASE_PORT",
                "HCCL_HOST_SOCKET_PORT_RANGE",
                "HCCL_NPU_SOCKET_PORT_RANGE",
            )
        )
    ):
        raise RuntimeError("CI smoke requires the complete dispatcher port lease and NODE_IPS")
    port = os.environ.get("MASTER_PORT")
    if nodes > 1 and not port:
        raise RuntimeError("multi-node smoke requires an explicit MASTER_PORT; no automatic port allocation")
    timeout = test.timeout or 7200
    if type(timeout) is not int or not 1 <= timeout <= 14400:
        raise ValueError("invalid execution timeout")
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(nodes),
        "--nproc-per-node",
        str(test.ngpu),
        "--node-rank",
        str(node_rank),
    ]
    if not port:
        cmd.append("--standalone")
    else:
        env["MASTER_ADDR"] = node_ips[0] if node_ips else "127.0.0.1"
        cmd.extend(["--master-addr", env["MASTER_ADDR"], "--master-port", port])
    cmd.extend(["-m", "tests.integration_tests.training_cases.dispatch_smoke"])
    print("[DISPATCH_SMOKE_LAUNCH] " + " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, env=env, check=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"dispatch smoke exceeded {timeout}-second execution budget") from exc
