import os
import sys
import json
import asyncio
import heapq
import subprocess
from typing import List, Optional, Any

import httpx
from vllm.logger import init_logger

logger = init_logger(__name__)


class VLLMInstance:

    def __init__(
        self,
        instance_type: str,
        device_id: int,
        host: str,
        port: int,
        model_path: str,
        instance_type_class: Any,
        ec_shared_storage_path: str,
        kv_port: Optional[int] = None,
        engine_id: Optional[int] = None,
    ):
        self.instance_type = instance_type
        self.device_id = device_id
        self.host = host
        self.port = port
        self.model_path = model_path
        self.process: Optional[subprocess.Popen] = None
        self.InstanceType = instance_type_class
        self.ec_shared_storage_path = ec_shared_storage_path
        self.kv_port = kv_port
        self.engine_id = engine_id

    def _ec_config(self, role: str) -> str:
        ec_cfg = {
            "ec_connector": "ECExampleConnector",
            "ec_role": role,
            "ec_connector_extra_config": {"shared_storage_path": self.ec_shared_storage_path},
        }
        return json.dumps(ec_cfg)

    def _kv_config(self, role: str) -> str:
        kv_cfg = {
            "kv_connector": "MooncakeLayerwiseConnector",
            "kv_role": role,
            "kv_port": str(self.kv_port),
            "engine_id": str(self.engine_id),
            "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
            "kv_connector_extra_config": {
                "use_ascend_direct": True,
                "prefill": {"dp_size": 1, "tp_size": 1},
                "decode": {"dp_size": 1, "tp_size": 1},
            },
        }
        return json.dumps(kv_cfg)

    def start(self, args) -> None:
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(self.device_id)

        cmd = [
            "vllm",
            "serve",
            self.model_path,
            "--port",
            str(self.port),
            "--enforce-eager",
            "--enable-request-id-headers",
            "--served-model-name",
            args.served_model_name,
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--allowed-local-media-path",
            args.allowed_local_media_path,
        ]

        if self.instance_type == self.InstanceType.ENCODE:
            cmd += [
                "--gpu-memory-utilization",
                str(args.encoder_gpu_memory_utilization),
                "--no-enable-prefix-caching",
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--ec-transfer-config",
                self._ec_config("ec_producer"),
            ]
        elif self.instance_type == self.InstanceType.PREFILL:
            cmd += [
                "--gpu-memory-utilization",
                str(args.prefill_gpu_memory_utilization),
                "--ec-transfer-config",
                self._ec_config("ec_consumer"),
                "--kv-transfer-config",
                self._kv_config("kv_producer"),
            ]
        elif self.instance_type == self.InstanceType.DECODE:
            cmd += [
                "--gpu-memory-utilization",
                str(args.decode_gpu_memory_utilization),
                "--kv-transfer-config",
                self._kv_config("kv_consumer"),
            ]
        elif self.instance_type == self.InstanceType.PD:
            cmd += [
                "--gpu-memory-utilization",
                str(args.pd_gpu_memory_utilization),
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
            ]
        else:
            logger.error("Unsupported instance type: %s", self.instance_type)
            return

        logger.info("Starting %s on device %s port %s", self.instance_type, self.device_id, self.port)
        logger.info("Command: %s", " ".join(cmd))

        self.process = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)

    def stop(self) -> None:
        if self.process:
            logger.info("Stopping instance on port %s", self.port)
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class ClusterManager:

    def __init__(self, args, instance_type_class: Any):
        self.args = args
        self.InstanceType = instance_type_class
        self.available_devices = list(args.visible_devices)
        self.used_devices: set[int] = set()
        self.instances: List[VLLMInstance] = []

        self.next_encoder_port = args.encoder_port_start
        self.next_prefill_port = args.prefill_port_start
        self.next_decode_port = args.decoder_port_start
        self.next_pd_port = args.pd_port_start
        self.next_kv_port = args.kv_port_start
        self.next_engine_id = args.engine_id_start

        self.lock = asyncio.Lock()

    def _alloc_device(self) -> int:
        if not self.available_devices:
            raise ResourceWarning("No available devices to scale out!")
        device_id = self.available_devices.pop(0)
        self.used_devices.add(device_id)
        return device_id

    def _alloc_kv(self) -> tuple[int, int]:
        kv_port = self.next_kv_port
        engine_id = self.next_engine_id
        self.next_kv_port += 1
        self.next_engine_id += 1
        return kv_port, engine_id

    def spawn_instance(self, instance_type: str, action: str = "start") -> Optional[VLLMInstance]:
        try:
            device_id = self._alloc_device()
        except ResourceWarning as e:
            logger.warning(str(e))
            return None

        if instance_type == self.InstanceType.ENCODE:
            port = self.next_encoder_port
            self.next_encoder_port += 1
            kv_port, engine_id = None, None
        elif instance_type == self.InstanceType.PREFILL:
            port = self.next_prefill_port
            self.next_prefill_port += 1
            kv_port, engine_id = self._alloc_kv()
        elif instance_type == self.InstanceType.DECODE:
            port = self.next_decode_port
            self.next_decode_port += 1
            kv_port, engine_id = self._alloc_kv()
        elif instance_type == self.InstanceType.PD:
            port = self.next_pd_port
            self.next_pd_port += 1
            kv_port, engine_id = None, None
        else:
            logger.error("Unsupported instance type: %s", instance_type)
            return None

        inst = VLLMInstance(
            instance_type=instance_type,
            device_id=device_id,
            host="127.0.0.1",
            port=port,
            model_path=self.args.model_path,
            instance_type_class=self.InstanceType,
            ec_shared_storage_path=self.args.ec_shared_storage_path,
            kv_port=kv_port,
            engine_id=engine_id,
        )
        inst.start(self.args)
        self.instances.append(inst)
        return inst

    async def _wait_ready(self, server) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        for _ in range(self.args.max_waiting_retries):
            try:
                response = await server.client.get(endpoint, headers=headers)
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(self.args.waiting_retry_interval)
        return False

    async def start_initial_cluster(self, mode: str):
        from epd_scaling_load_balance_proxy import ServerState, DeploymentMode

        result = {}
        for type_name in DeploymentMode.ACTIVE_TYPES[mode]:
            inst = self.spawn_instance(type_name, action="init")
            if not inst:
                logger.error("Failed to start %s instance", type_name)
                sys.exit(1)
            server = ServerState(inst.host, inst.port)
            await self._wait_ready(server)
            result[type_name] = [server]
        return result

    async def monitor_and_scale(self, proxy_state, mode: str):
        from epd_scaling_load_balance_proxy import DeploymentMode, InstanceType, ServerState

        type_config = {
            InstanceType.ENCODE:   (proxy_state.encoders,   self.args.encode_threshold),
            InstanceType.PREFILL:  (proxy_state.prefillers,  self.args.prefill_threshold),
            InstanceType.DECODE:   (proxy_state.decoders,    self.args.decode_threshold),
            InstanceType.PD:       (proxy_state.pds,         self.args.pd_threshold),
        }
        active_types = DeploymentMode.ACTIVE_TYPES[mode]
        while True:
            await asyncio.sleep(self.args.scale_interval)
            for t in active_types:
                try:
                    servers, threshold = type_config[t]
                    await self._maybe_scale(proxy_state, t, servers, threshold, ServerState)
                    await self._cleanup_stopping(proxy_state, t, servers)
                except Exception as e:
                    logger.error("Error in autoscaling for %s: %s", t, e)

    def _set_instance_status(self, proxy_state, instance_type: str, server, status: str) -> None:
        server.status = status
        if instance_type == self.InstanceType.ENCODE:
            target_heap = proxy_state.encoder_heap
            all_instances = proxy_state.encoders
            priority_func = lambda s: s.active_tokens
        elif instance_type == self.InstanceType.PREFILL:
            target_heap = proxy_state.prefiller_heap
            all_instances = proxy_state.prefillers
            priority_func = lambda s: s.active_tokens + s.active_kv_cache * 0.3
        elif instance_type == self.InstanceType.DECODE:
            target_heap = proxy_state.decoder_heap
            all_instances = proxy_state.decoders
            priority_func = lambda s: s.active_tokens
        elif instance_type == self.InstanceType.PD:
            target_heap = proxy_state.pd_heap
            all_instances = proxy_state.pds
            priority_func = lambda s: s.active_tokens
        else:
            logger.error("Unsupported instance type: %s", instance_type)
            return

        if status == "stopping":
            new_heap = [item for item in target_heap if item[2] != server]
            heapq.heapify(new_heap)
            if instance_type == self.InstanceType.ENCODE:
                proxy_state.encoder_heap = new_heap
            elif instance_type == self.InstanceType.PREFILL:
                proxy_state.prefiller_heap = new_heap
            elif instance_type == self.InstanceType.DECODE:
                proxy_state.decoder_heap = new_heap
            elif instance_type == self.InstanceType.PD:
                proxy_state.pd_heap = new_heap

            logger.info("Instance %s marked STOPPING; removed from dispatch queue.", server)
        elif status == "working":
            in_heap = any(item[2] == server for item in target_heap)
            if not in_heap:
                try:
                    current_idx = all_instances.index(server)
                    heapq.heappush(target_heap, (priority_func(server), current_idx, server))
                    logger.info("Instance %s resurrected to WORKING.", server)
                except ValueError:
                    logger.error("Cannot resurrect %s: not found in proxy instance list.", server)

    async def _cleanup_stopping(self, proxy_state, instance_type: str, servers):
        for server in list(servers):
            if server.status != "stopping":
                continue
            if server.active_tokens > 0.1:
                continue
            if instance_type == self.InstanceType.PREFILL and server.active_kv_cache > 0.1:
                continue
            await server.client.aclose()
            await proxy_state.remove_instances(instance_type, [server])
            self._kill_process_by_port(server.port)

    async def _maybe_scale(self, proxy_state, instance_type: str, servers, threshold: int, server_cls):
        if not servers:
            return

        working = [s for s in servers if s.status == "working"]
        stopping = [s for s in servers if s.status == "stopping"]
        if not working:
            return

        tokens = [s.active_tokens for s in working]
        min_tokens = min(tokens)
        max_tokens = max(tokens)
        scale_in_threshold = threshold * self.args.scale_in_ratio

        if min_tokens >= threshold:
            async with self.lock:
                if stopping:
                    self._set_instance_status(proxy_state, instance_type, stopping[0], "working")
                else:
                    new_inst = self.spawn_instance(instance_type, action="scale_out")
                    if not new_inst:
                        return
                    server = server_cls(new_inst.host, new_inst.port)
                    await self._wait_ready(server)
                    await proxy_state.add_instances(instance_type, [server])
        elif max_tokens < scale_in_threshold and len(working) > 1:
            victim = min(working, key=lambda s: s.active_tokens)
            self._set_instance_status(proxy_state, instance_type, victim, "stopping")

    def _kill_process_by_port(self, port: int):
        target_inst = None
        for inst in self.instances:
            if inst.port == port:
                target_inst = inst
                break
        if target_inst:
            logger.info("Stopping process on port %s, releasing device %s", port, target_inst.device_id)
            target_inst.stop()
            self.instances.remove(target_inst)
            self.available_devices.append(target_inst.device_id)
            self.available_devices.sort()
            if target_inst.device_id in self.used_devices:
                self.used_devices.remove(target_inst.device_id)
