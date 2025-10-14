import os
from contextlib import contextmanager

import torch.distributed as dist


@contextmanager
def temp_env(env_dict):
    """
    临时设置环境变量，上下文退出时恢复原始值
    """
    old_env = {}
    for k, v in env_dict.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                # 之前没有这个变量，就删除
                os.environ.pop(k, None)
            else:
                # 恢复原来的值
                os.environ[k] = v


@contextmanager
def dist_group(backend="gloo"):
    """
    上下文管理器：初始化分布式进程组，退出时自动销毁
    """
    if dist.is_initialized():
        yield
        return

    dist.init_process_group(backend=backend)
    try:
        yield
    finally:
        dist.destroy_process_group()


def sync_node_roles(with_prefill: bool, headless: bool):
    """
    在所有节点之间同步 (with_prefill, headless) 信息
    """
    world_size = dist.get_world_size()
    local_info = (with_prefill, headless)
    gathered_info = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_info, local_info)
    return gathered_info


if __name__ == "__main__":
    # 假设我们知道节点 IP
    ips = ["10.0.0.148", "10.0.0.142"]
    cur_ip = os.popen("hostname -I").read().strip().split()[0]  # 获取当前节点 IP

    env_vars = {
        "MASTER_ADDR": ips[0],
        "MASTER_PORT": 29500,
        "WORLD_SIZE": len(ips),
        "RANK": ips.index(cur_ip)
    }

    with temp_env(env_vars):
        print(
            f"[{cur_ip}] Temp env set. Rank={os.environ['RANK']}, WorldSize={os.environ['WORLD_SIZE']}"
        )

        with dist_group("gloo"):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f"[{cur_ip}] Connected. Rank {rank}/{world_size}")

            # 每个节点定义自己的状态
            with_prefill = os.getenv("WITH_PREFILL", "1") == "1"
            headless = os.getenv("HEADLESS", "0") == "1"

            # 同步所有节点状态
            all_roles = sync_node_roles(with_prefill, headless)

            if rank == 0:
                print("=== All Node Roles ===")
                for i, (p, h) in enumerate(all_roles):
                    print(f"Node {i}: with_prefill={p}, headless={h}")

    # 上下文退出后，MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE 都会恢复/删除
    print(
        f"[{cur_ip}] Temp env cleaned. MASTER_ADDR={os.environ.get('MASTER_ADDR')}"
    )
