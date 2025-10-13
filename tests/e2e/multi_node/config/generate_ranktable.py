import logging
import os
import socket
import subprocess
from pathlib import Path

from tests.e2e.multi_node.config.common import (ASCEND_ENV_PATH,
                                                DISAGGEGATED_PREFILL_PORT,
                                                LIB_PATH, RANKTABLE_GEN_PATH,
                                                RANKTABLE_PATH)
from tests.e2e.multi_node.config.utils import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def setup_and_run_ranktable(
    ips: list[str],
    npus_per_node: int = 16,
    network_card_name: str = None,
    prefill_device_cnt: int = None,
    decode_device_cnt: int = None,
    gen_ranktable=True,
):
    """Generate ranktable.json for multi-node setup.
    Args:
        ips (list[str]): ips for all nodes.
        npus_per_node (int): num of NPUs per node.
        network_card_name (str): network card name (e.g., eth0).
        prefill_device_cnt (int): num of devices for prefill stage.
        decode_device_cnt (int): num of devices for decode stage.
        gen_ranktable (bool): whether to generate ranktable.json.
    """

    if Path(RANKTABLE_PATH).exists():
        logger.info(
            f"RANKTABLE_PATH is already set: {os.getenv('RANKTABLE_PATH')}")
        return

    # === setup env ===
    ascend_env = ASCEND_ENV_PATH
    if os.path.exists(ascend_env):
        subprocess.run(["bash", "-c", f"source {ascend_env}"], check=False)
    else:
        logger.warning(f"Ascend env file not found: {ascend_env}")

    lib_path = LIB_PATH
    os.environ[
        "LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    local_ips = socket.gethostbyname_ex(socket.gethostname())[2]
    local_host = "127.0.0.1"
    master_addr = ips[0]
    master_port = DISAGGEGATED_PREFILL_PORT
    nnodes = len(ips)
    node_rank = None

    for i, ip in enumerate(ips):
        if ip in local_ips:
            local_host = ip
            node_rank = i
            break

    if node_rank is None:
        logger.error(
            '"NODE_RANK" must be defined — local host not in provided IP list')
        raise ValueError("Local host IP not found in ips list")

    world_size = npus_per_node * nnodes
    rank_start = npus_per_node * node_rank

    logger.info("\n========> Parameters:\n"
                f"LOCAL_HOST: {local_host}\n"
                f"WORLD_SIZE: {world_size}\n"
                f"RANKSTART: {rank_start}\n"
                f"NNODES: {nnodes}\n"
                f"NODE_RANK: {node_rank}\n"
                "=====================")

    ranktable_path = Path(RANKTABLE_PATH)

    if gen_ranktable or not os.path.exists(ranktable_path):
        cmd = [
            "torchrun",
            "--nproc_per_node",
            "1",
            "--nnodes",
            str(nnodes),
            "--node_rank",
            str(node_rank),
            "--master_addr",
            master_addr,
            "--master_port",
            str(master_port),
            RANKTABLE_GEN_PATH,
            "--local-host",
            local_host,
            "--prefill-device-cnt",
            str(prefill_device_cnt),
            "--decode-device-cnt",
            str(decode_device_cnt),
        ]

        env = os.environ.copy()
        if network_card_name:
            env["GLOO_SOCKET_IFNAME"] = network_card_name

        logger.info("Running command:")
        logger.info(" ".join(cmd))

        subprocess.run(cmd, env=env, check=True)
    else:
        logger.info(
            f"Skip generating ranktable: {ranktable_path} already exists.")


if __name__ == "__main__":
    setup_and_run_ranktable(
        ips=["10.0.0.158", "10.0.0.143"],
        npus_per_node=16,
        network_card_name="eth0",
        prefill_device_cnt=16,
        decode_device_cnt=16,
        gen_ranktable=True,
    )
