import os
import time

os.environ["RAY_DEDUP_LOGS"] = "0"
import asyncio
import threading
from contextlib import suppress
from typing import Any

import ray
import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from worm_server.worm_servlet import WORMServlet

BODY = Body(...)

# NEWFORMWARE SERVER SPECS
os.environ["MASTER_ADDR"] = os.environ["HEAD_IP"]
os.environ["MASTER_PORT"] = "44551"
os.environ["HEAD_IFACE"] = "enp23s0f3"
os.environ["WORKER_IFACE"] = "enp23s0f3"

WORLD_SIZE = int(os.environ["WORLD_SIZE"])
MODEL_SIZE = int(os.environ["MODEL_SIZE"])
TP_SIZE = int(os.environ["TP_SIZE"])
NPU_START_IDX = int(os.environ["NPU_START_IDX"])
PORT_NUM = int(os.environ["PORT_NUM"])


class WormServer:
    def __init__(
        self,
        world_size: int,
        model_size: int,
        tensor_parallel_size: int,
        api_port: int = 8000,
        actor_name_fmt: str = "actor-{}",
    ):
        self.world_size = world_size
        self.tensor_parallel_size = tensor_parallel_size
        self.model_size = model_size
        self.api_port = api_port
        self.actor_name_fmt = actor_name_fmt

        try:
            self.MODEL_PATH = os.environ["MODEL_PATH"]
        except KeyError as exc:
            raise RuntimeError("MODEL_PATH env var not set") from exc

        self.MASTER_ADDR = os.environ["MASTER_ADDR"]
        self.MASTER_PORT = os.environ["MASTER_PORT"]
        self.HEAD_IP = os.environ["HEAD_IP"]
        self.HEAD_IFACE = os.environ["HEAD_IFACE"]
        self.WORKER_IFACE = os.environ["WORKER_IFACE"]

        self._app = None
        self._server_thread = None
        self.actors = []  # list[ray.actor.ActorHandle]
        # Ensure Ray is up
        ray.init(address="auto")
        nic_for_node, plan = self._create_placement_plan()
        # Build placement plan and create actors
        self._create_actors_in_order(nic_for_node, plan)
        print("node map:", nic_for_node)
        print("cluster resources:", ray.cluster_resources())

        # Diagnostics
        print(f"health() -> {ray.get([a.health.remote() for a in self.actors])}")
        print(f"peek_env()\n{sorted(ray.get([a.peek_env.remote() for a in self.actors]), key=lambda x: x['rank'])}")

        # Initializing HCCL and loading the model
        print(f"_init_global_world -> {ray.get([a._init_global_world.remote() for a in self.actors])}")
        print(f"warm_up -> {ray.get([a.warm_up.remote() for a in self.actors])}")
        # print(f'ring_probe -> {ray.get([a.ring_probe.remote() for a in self.actors])}')
        # print(f'_init_model_world() {ray.get([a._init_model_world.remote() for a in self.actors])}')
        print(f"_init_model() {ray.get([a._init_model.remote() for a in self.actors[: self.tensor_parallel_size]])}")
        gpu_blocks = int(os.getenv("KV_GPU_BLOCKS"))
        results = ray.get(
            [
                actor._init_kv_cache.remote(num_gpu_blocks=gpu_blocks)
                for actor in self.actors[: self.tensor_parallel_size]
            ]
        )
        print(f"_init_kv_cache() {results}")

        results = ray.get(
            [
                actor.set_attribute.remote(
                    name="peers",
                    value_or_oid=self.actors,
                )
                for actor in self.actors
            ]
        )
        print(f"set_attribute(peers) -> {results}")

        print(f"broadcast_metadata_dict(peers) -> {ray.get(ray.get(self.actors[0].broadcast_metadata_dict.remote()))}")

        # Scale up to get to the initial config
        num_npus = self.model_size - self.tensor_parallel_size
        results = ray.get([actor.scaleup.remote(num_npus=num_npus) for actor in self.actors])
        print(f"scaleup() -> {results}")
        print(f"model_health() -> {ray.get([a.model_health.remote() for a in self.actors])}")

        # API app
        self._build_app()

    def _create_placement_plan(self):
        nodes = [n for n in ray.nodes() if n["Alive"]]
        nic_for_node = {}
        for n in nodes:
            node_id = n["NodeID"]
            ip = n["NodeManagerAddress"]
            iface = self.HEAD_IFACE if ip == self.HEAD_IP else self.WORKER_IFACE
            res = n.get("Resources", {})
            capacity = int(res.get("NPU") or 0)
            nic_for_node[node_id] = {"ip": ip, "iface": iface, "capacity": capacity}

        if not nic_for_node:
            raise RuntimeError("No alive Ray nodes found")
        head_node_id = next((k for k, v in nic_for_node.items() if v["ip"] == self.HEAD_IP), None)
        if head_node_id is None:
            raise RuntimeError(f"HEAD_IP={self.HEAD_IP} not found among Ray nodes: {nic_for_node}")
        nodes_in_order = [(head_node_id, nic_for_node[head_node_id]["capacity"])]
        for nid in sorted([k for k in nic_for_node if k != head_node_id], key=lambda x: nic_for_node[x]["ip"]):
            nodes_in_order.append((nid, nic_for_node[nid]["capacity"]))

        npu_start_idx = NPU_START_IDX
        plan = []
        r = 0
        for nid, cap in nodes_in_order:
            for local_idx in range(npu_start_idx, cap + npu_start_idx):
                if r >= self.world_size:
                    break
                plan.append((nid, local_idx))
                r += 1
            if r >= self.world_size:
                break
        if len(plan) < self.world_size:
            raise RuntimeError(
                f"""Insufficient NPUs for WORLD_SIZE={self.world_size}. 
                Available {len(plan)} across nodes: {nic_for_node}"""
            )
        assert nic_for_node[head_node_id]["ip"] == self.MASTER_ADDR, (
            f"Placement bug: rank-0 isn't on MASTER_ADDR={self.MASTER_ADDR}"
        )

        return nic_for_node, plan

    def _create_actors_in_order(self, nic_for_node, plan):
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        actors = []
        for r in range(self.world_size):
            nid, local_npu_idx = plan[r]
            nic = nic_for_node[nid]
            a = WORMServlet.options(
                name=self.actor_name_fmt.format(r),
                resources={"NPU": 1},  # uses Ray's 'NPU' resource accounting
                runtime_env={
                    "env_vars": {
                        "MODEL_PATH": self.MODEL_PATH,
                        "MASTER_ADDR": self.MASTER_ADDR,
                        "MASTER_PORT": self.MASTER_PORT,
                        "HCCL_IF_IP": nic["ip"],
                        "GLOO_SOCKET_IFNAME": nic["iface"],
                        "HCCL_CONNECT_TIMEOUT": "1200",
                        "HCCL_WHITELIST_DISABLE": "1",
                        "VLLM_VERSION": os.getenv("VLLM_VERSION", None),
                        "VLLM_USE_V1": os.getenv("VLLM_USE_V1", None),
                        "HCCL_HOST_SOCKET_PORT_RANGE": "60100-60150",
                        "HCCL_NPU_SOCKET_PORT_RANGE": "60100-60150",
                        "BLOCK_SIZE": os.getenv("BLOCK_SIZE", 16),
                        "NUM_MODEL_LAYERS": os.getenv("NUM_MODEL_LAYERS", "0"),
                        "PYTHONPATH": os.getenv("PYTHONPATH"),
                        "KV_GPU_BLOCKS": os.getenv("KV_GPU_BLOCKS", "0"),
                    }
                },
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False),
            ).remote(
                rank=r,
                global_world_size=self.world_size,
                tensor_parallel_size=self.tensor_parallel_size,
                node_id=nid,  # correspondingIP is in env HCCL_IF_IP
                phy_npu_id=local_npu_idx,
            )
            actors.append(a)
        self.actors = actors

    def _create_placement_plan_for(self, target_world_size: int):
        nodes = [n for n in ray.nodes() if n["Alive"]]
        nic_for_node = {}
        for n in nodes:
            node_id = n["NodeID"]
            ip = n["NodeManagerAddress"]
            iface = self.HEAD_IFACE if ip == self.HEAD_IP else self.WORKER_IFACE
            res = n.get("Resources", {})
            capacity = int(res.get("NPU") or 0)
            nic_for_node[node_id] = {"ip": ip, "iface": iface, "capacity": capacity}

        head_node_id = next((k for k, v in nic_for_node.items() if v["ip"] == self.HEAD_IP), None)

        nodes_in_order = [(head_node_id, nic_for_node[head_node_id]["capacity"])]
        for nid in sorted([k for k in nic_for_node if k != head_node_id], key=lambda x: nic_for_node[x]["ip"]):
            nodes_in_order.append((nid, nic_for_node[nid]["capacity"]))

        npu_start_idx = NPU_START_IDX
        plan = []
        r = 0
        for nid, cap in nodes_in_order:
            for local_idx in range(npu_start_idx, cap + npu_start_idx):
                if r >= target_world_size:
                    break
                plan.append((nid, local_idx))
                r += 1
            if r >= target_world_size:
                break

        if len(plan) < target_world_size:
            raise RuntimeError(f"Insufficient NPUs: need {target_world_size}, planned {len(plan)}")
        return nic_for_node, plan

    def _create_actors_range(self, nic_for_node, plan, start_rank, end_rank, target_world_size):
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        new_actors = []
        for r in range(start_rank, end_rank):
            nid, local_npu_idx = plan[r]
            nic = nic_for_node[nid]
            print(f"HURR {os.getenv('VLLM_VERSION', None)}")
            a = WORMServlet.options(
                name=self.actor_name_fmt.format(r),
                resources={"NPU": 1},
                runtime_env={
                    "env_vars": {
                        "MODEL_PATH": self.MODEL_PATH,
                        "MASTER_ADDR": self.MASTER_ADDR,
                        "MASTER_PORT": self.MASTER_PORT,
                        "HCCL_IF_IP": nic["ip"],
                        "GLOO_SOCKET_IFNAME": nic["iface"],
                        "HCCL_CONNECT_TIMEOUT": "1200",
                        "HCCL_WHITELIST_DISABLE": "1",
                        "VLLM_VERSION": os.getenv("VLLM_VERSION", None),
                        "VLLM_USE_V1": os.getenv("VLLM_USE_V1", None),
                    }
                },
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False),
            ).remote(
                rank=r,
                global_world_size=target_world_size,
                tensor_parallel_size=self.tensor_parallel_size,
                node_id=nid,
                phy_npu_id=local_npu_idx,
            )
            new_actors.append(a)
        return new_actors

    def add_npus(self, num_npus: int):
        assert num_npus > 0
        old_world = self.world_size
        target_world = old_world + num_npus

        # 1) Build placement for the target world
        nic_for_node, plan = self._create_placement_plan_for(target_world)

        # 2) Create new actors for ranks [old_world, target_world)
        new_actors = self._create_actors_range(
            nic_for_node, plan, old_world, target_world, target_world_size=target_world
        )

        # (Optional warm boot wait)
        ray.get([a.health.remote() for a in new_actors])

        # 3) Destroy old distributed world on ALL actors (old + new)
        all_actors = self.actors + new_actors
        ray.get([a._destroy_global_world.remote() for a in all_actors])

        # 4) Update global_world_size on ALL actors; peers, too
        _ = ray.get([a.set_attribute.remote("global_world_size", target_world) for a in all_actors])
        _ = ray.get([a.set_attribute.remote("peers", all_actors) for a in all_actors])
        print(f"broadcast_metadata_dict(peers) -> {ray.get(ray.get(self.actors[0].broadcast_metadata_dict.remote()))}")

        # 5) Re-init distributed world on ALL actors
        init_results = ray.get([a._init_global_world.remote() for a in all_actors])

        # 6) Quick smoke test
        _ = ray.get([a.warm_up.remote() for a in all_actors])

        # # 7) Test Scale Up
        print(f"scaleup() -> {ray.get([a.scaleup.remote(num_npus=2) for a in all_actors])}")
        print(f"model_health() -> {ray.get([a.model_health.remote() for a in all_actors])}")

        # Commit locally
        self.world_size = target_world
        self.actors = all_actors

        return {
            "old_world": old_world,
            "new_world": target_world,
            "init": init_results,
        }

    # ========== FastAPI ==========
    async def _gather(self, futures):
        return await asyncio.to_thread(ray.get, futures)

    def _build_app(self):
        app = FastAPI(title="WORM Control API", version="1.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/health")
        async def health():
            futs = [a.health.remote() for a in self.actors]
            return {"actors": await self._gather(futs)}

        @app.get("/model_health")
        async def model_health():
            futs = [a.model_health.remote() for a in self.actors]
            return {"actors": await self._gather(futs)}

        @app.post("/scaleup")
        async def scaleup(num_npus: int = 2):
            futs = [a.scaleup.remote(num_npus=num_npus) for a in self.actors]
            return {"result": await self._gather(futs)}

        @app.post("/scaledown")
        async def scaledown(num_npus: int = 2):
            futs = [a.scaledown.remote(num_npus=num_npus) for a in self.actors]
            return {"result": await self._gather(futs)}

        @app.post("/addnpus")
        async def addnpus(num_npus: int):
            # run the blocking orchestration off the event loop
            result = await asyncio.to_thread(self.add_npus, num_npus)
            return {"result": result}

        @app.post("/invoke_method")
        async def invoke_method(
            method_name: str,
            payload: dict[str, Any] = BODY,
        ):
            if payload is None:
                payload = {}
            args = payload.get("args", []) or []
            kwargs = payload.get("kwargs", {}) or {}
            futs = [getattr(a, method_name).remote(*args, **kwargs) for a in self.actors]
            results = await self._gather(futs)
            return {"method": method_name, "results": results}

        @app.get("/cluster_status")
        async def cluster_status():
            return {
                "nodes": [
                    {
                        "node_id": n["NodeID"],
                        "ip": n["NodeManagerAddress"],
                        "alive": n["Alive"],
                        "resources": n.get("Resources", {}),
                    }
                    for n in ray.nodes()
                ],
                "cluster_resources": ray.cluster_resources(),
                "available_resources": ray.available_resources(),
            }

        self._app = app

    def start(self, host: str = "0.0.0.0", log_level: str = "info"):
        if self._app is None:
            raise RuntimeError("API app not built")

        def _run():
            uvicorn.run(self._app, host=host, port=self.api_port, log_level=log_level)

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        print(f"HTTP API listening on :{self.api_port}")

    def stop(self):
        with suppress(Exception):
            ray.get([a.free.remote() for a in self.actors])


if __name__ == "__main__":
    # launch_ray_cluster(num_npus=8)  # optional if you auto-attach to an existing cluster
    svc = WormServer(world_size=WORLD_SIZE, model_size=MODEL_SIZE, tensor_parallel_size=TP_SIZE, api_port=PORT_NUM)
    svc.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
