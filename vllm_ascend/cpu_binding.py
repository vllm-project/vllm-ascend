#!/usr/bin/env python3

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import psutil
import regex as re
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True, kw_only=True)
class CPUBindingDeployment:
    """Pre-fork facts, copied with VllmConfig; never reconstructed by workers."""

    sizes: tuple[int, int, int]
    dp_size_local: int
    dp_start_rank: int
    devices: tuple[int, ...]
    allowed_cpus: tuple[int, ...]
    reason: str | None


def prepare_cpu_binding(vllm_config: VllmConfig) -> None:
    """Capture once in the platform hook, including unsupported deployments."""
    if getattr(vllm_config, "_ascend_cpu_binding_deployment", None) is not None:
        return
    parallel = vllm_config.parallel_config
    sizes = (parallel.data_parallel_size, parallel.pipeline_parallel_size, parallel.tensor_parallel_size)
    backend = parallel.distributed_executor_backend
    dp_backend = getattr(parallel, "data_parallel_backend", "mp")
    external_dp = bool(
        getattr(parallel, "data_parallel_external_lb", False)
        or getattr(parallel, "data_parallel_hybrid_lb", False)
        or (sizes[0] > 1 and parallel.data_parallel_size_local != sizes[0])
    )
    reason = None
    if parallel.nnodes != 1:
        reason = "multi-node deployment"
    elif backend not in ("uni", "mp") or dp_backend != "mp":
        reason = "requires uni/mp with the mp DP backend"
    elif parallel.prefill_context_parallel_size != 1:
        reason = "PCP is not supported"
    elif parallel.enable_elastic_ep:
        reason = "elastic membership is not supported"
    elif external_dp:
        reason = "independently launched DP membership is not supported"
    devices: tuple[int, ...] = ()
    allowed_cpus: tuple[int, ...] = ()
    if reason is None:
        try:
            worker_count = sizes[0] * sizes[1] * sizes[2]
            assigned_devices = getattr(parallel, "assigned_physical_gpu_ids", None)
            if assigned_devices is not None:
                # These IDs are already physical. Capture their order before DP
                # sharding; the platform's global mapping is only set in workers.
                devices = tuple(assigned_devices[:worker_count])
            else:
                # Include participating slots only, preserving visibility order.
                devices = tuple(
                    current_platform.device_id_to_physical_device_id(index) for index in range(worker_count)
                )
            if len(devices) != worker_count:
                raise ValueError(f"expected {worker_count} assigned devices, got {len(devices)}")
            if len(set(devices)) != worker_count or any(device < 0 for device in devices):
                raise ValueError("participating physical device IDs must be unique and non-negative")
            allowed_cpus = tuple(sorted(os.sched_getaffinity(0)))
        except (ValueError, IndexError, OSError) as exc:
            reason = f"cannot capture shared CPU/device layout: {exc}"
    vllm_config._ascend_cpu_binding_deployment = CPUBindingDeployment(
        sizes=sizes,
        dp_size_local=parallel.data_parallel_size_local,
        dp_start_rank=parallel.data_parallel_rank - (parallel.data_parallel_rank_local or 0),
        devices=devices,
        allowed_cpus=allowed_cpus,
        reason=reason,
    )


def _bind_msboost(vllm_config: VllmConfig | None, npu_id: int, migrate_memory: bool) -> None:
    """Optional worker-only import; runtime failures must not stop inference."""
    deployment = getattr(vllm_config, "_ascend_cpu_binding_deployment", None)
    if deployment is None:
        logger.warning("[cpu_binding.msboost.skipped] missing pre-fork deployment context")
        return
    if deployment.reason:
        logger.warning("[cpu_binding.msboost.skipped] %s", deployment.reason)
        return
    try:
        from affinity_bind.api import apply_vllm_affinity
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        # A configuration object enables binding and overrides component defaults.
        # Let affinity_bind validate its own tuning keys instead of duplicating its schema.
        binding_config = (vllm_config.additional_config or {}).get("enable_cpu_binding", True)
        options = {"strategy": "auto"}
        if isinstance(binding_config, dict):
            options.update(binding_config)
        coordinate = (
            vllm_config.parallel_config.data_parallel_index,
            get_pp_group().rank_in_group,
            get_tp_group().rank_in_group,
        )
        dp_size, pp_size, tp_size = deployment.sizes
        result = apply_vllm_affinity(
            dp_size=dp_size,
            dp_size_local=deployment.dp_size_local,
            dp_start_rank=deployment.dp_start_rank,
            tp_size=tp_size,
            pp_size=pp_size,
            coordinate=coordinate,
            device=npu_id,
            migrate_memory=migrate_memory,
            options=options,
            logger=logger,
            devices=deployment.devices,
            allowed_cpus=deployment.allowed_cpus,
        )
        if result.get("status") != "completed":
            logger.warning("[cpu_binding.msboost.incomplete] coordinate=%s result=%s", coordinate, result)
        else:
            logger.info("[cpu_binding.msboost.done] coordinate=%s result=%s", coordinate, result)
    except Exception:
        logger.warning(
            "[cpu_binding.msboost.failed] device=%s; affinity may be partially applied; "
            "continuing without native fallback",
            npu_id,
            exc_info=True,
        )


MASK_BIT = 32  # Number of bits in a CPU affinity mask group
MIN_CPUS_PER_NPU = 5  # 2(IRQ) + 1(main, at least 1 CPU) + 1(acl) + 1(release) = 5 CPUs per NPU
ALLOWED_CPUS_PATH = "/proc/self/status"
ASCEND_RT_VISIBLE_DEVICES = os.getenv("ASCEND_RT_VISIBLE_DEVICES")


def is_arm_cpu() -> bool:
    arch = platform.machine().lower()
    if arch in {"x86_64", "amd64", "i386", "i686"}:
        return False
    if arch in {"aarch64", "arm64"} or arch.startswith("arm"):
        return True
    logger.warning(
        "Unknown CPU architecture. arch=%s, action: disabling CPU binding.",
        arch,
    )
    return False


def execute_command(cmd: list[str]) -> tuple[str, int]:
    # Force a C locale so subprocess output remains parseable on localized OSes.
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["LC_MESSAGES"] = "C"
    with subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) as p:
        try:
            out, _ = p.communicate(timeout=1000)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
    return out.decode(), p.returncode


class DeviceInfo:
    def __init__(self):
        self.npu_map_info: dict[str, dict[str, str]] = self.get_npu_map_info()
        self.allowed_cpus: list[int] = self.parse_allowed_cpus()
        self.running_npu_list: list[int] = self.get_running_npus()
        self.npu_affinity: dict[int, list[int]] = self.parse_topo_affinity()
        self.all_logic_npus: list[int] = self.get_all_logic_npus()
        self.total_logic_npus: int = len(self.all_logic_npus)

    @staticmethod
    def split_npu_smi_header(line: str) -> list[str]:
        return [item.strip() for item in re.split(r"\s{2,}", line.strip()) if item.strip()]

    @staticmethod
    def is_cpu_list(cpu_list_str: str) -> bool:
        return bool(re.fullmatch(r"\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*", cpu_list_str))

    @staticmethod
    def expand_cpu_list(allowed_list_str: str) -> list[int]:
        allowed_cpus_list: list[int] = []
        for per_range in allowed_list_str.split(","):
            if "-" in per_range:
                start_cpu, end_cpu = map(int, per_range.split("-"))
                allowed_cpus_list.extend(range(start_cpu, end_cpu + 1))
            else:
                allowed_cpus_list.append(int(per_range))
        return allowed_cpus_list

    def get_all_logic_npus(self) -> list[int]:
        """Collect all logical NPU IDs from the NPU mapping.

        The per-chip map uses chip_id as the key and the logical NPU ID
        string as the value.
        """
        logic_ids: set[int] = set()
        for _, chip_map in self.npu_map_info.items():
            for _, logic_str in chip_map.items():
                if logic_str and logic_str.isdigit():
                    logic_ids.add(int(logic_str))
        return sorted(logic_ids)

    @staticmethod
    def get_npu_map_info() -> dict[str, dict[str, str]]:
        npu_map_info: dict[str, dict[str, str]] = {}
        npu_info, _ = execute_command(["npu-smi", "info", "-m"])
        npu_map = [line.strip() for line in npu_info.splitlines() if line.strip()]

        header = DeviceInfo.split_npu_smi_header(npu_map[0])
        npu_id_idx = header.index("NPU ID")
        chip_id_idx = header.index("Chip ID") if "Chip ID" in header else None
        chip_logic_id_idx = header.index("Chip Logic ID") if "Chip Logic ID" in header else None

        for line in npu_map[1:]:
            parts = line.split()
            npu_id = parts[npu_id_idx]

            chip_id = "0"
            if chip_id_idx is not None:
                chip_id = parts[chip_id_idx]

            if chip_logic_id_idx is not None:
                chip_logic_id = parts[chip_logic_id_idx]
            else:
                chip_logic_id = npu_id
            if not chip_logic_id.isdigit():
                continue
            if npu_id not in npu_map_info:
                npu_map_info[npu_id] = {}
            npu_map_info[npu_id][chip_id] = chip_logic_id
        return npu_map_info

    def resolve_logic_id(self, npu_id: str, chip_id: str | None) -> int:
        chip_map = self.npu_map_info.get(npu_id, {})
        if chip_id is not None:
            chip_logic_id = chip_map.get(chip_id)
        elif len(chip_map) == 1:
            chip_logic_id = next(iter(chip_map.values()))
        else:
            raise RuntimeError(
                "Failed to resolve chip_logic_id because the process table does not contain chip_id "
                f"and NPU {npu_id} has {len(chip_map)} mapped chips."
            )
        if not chip_logic_id or not chip_logic_id.isdigit():
            raise RuntimeError("Failed to get correct chip_logic_id from command 'npu-smi info -m'.")
        return int(chip_logic_id)

    def get_running_npus(self) -> list[int]:
        npu_message, _ = execute_command(["npu-smi", "info"])
        in_proc_section = False
        proc_npu_field = ""
        running_npu_set = set()
        for line in npu_message.splitlines():
            line = line.strip()
            if line.startswith("| NPU") and "Process id" in line:
                parts = [p.strip() for p in line.strip("|").split("|")]
                proc_npu_field = " ".join(parts[0].split())
                in_proc_section = True
                continue
            if not in_proc_section:
                continue
            if line.startswith("| "):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) < 2:
                    continue
                npu_chip_parts = parts[0].split()
                npu_id = npu_chip_parts[0]
                chip_id = None
                if proc_npu_field == "NPU Chip":
                    chip_id = npu_chip_parts[1]
                if not npu_id.isdigit() or (chip_id is not None and not chip_id.isdigit()):
                    continue
                running_npu_set.add(self.resolve_logic_id(npu_id, chip_id))
        if ASCEND_RT_VISIBLE_DEVICES:
            devices_str = ASCEND_RT_VISIBLE_DEVICES
            devices_list = [int(x) for x in devices_str.split(",")]
            running_npu_set = set(devices_list) & running_npu_set
        if not running_npu_set:
            raise RuntimeError("Can not get running npu info.")
        return sorted(running_npu_set)

    def parse_allowed_cpus(self) -> list[int]:
        if not os.path.exists(ALLOWED_CPUS_PATH):
            return []
        with open(ALLOWED_CPUS_PATH) as f:
            for line in f:
                if line.startswith("Cpus_allowed_list"):
                    return self.expand_cpu_list(line.split()[1])
        raise RuntimeError("Can not found specific 'Cpus_allowed_list' in the '/proc/self/status' file.")

    def parse_topo_affinity(self) -> dict[int, list[int]]:
        affinity: dict[int, list[int]] = {}
        affinity_message, _ = execute_command(["npu-smi", "info", "-t", "topo"])
        for line in affinity_message.splitlines():
            if line.startswith("NPU"):
                parts = line.split()
                npu_match = re.fullmatch(r"NPU(\d+)", parts[0])
                if not npu_match:
                    continue
                last_part = parts[-1]
                if self.is_cpu_list(last_part):
                    affinity[int(npu_match.group(1))] = self.expand_cpu_list(last_part)
        return affinity


class CpuAlloc:
    def __init__(self, rank_id: int, npu_id: int):
        self.rank_id = rank_id
        self.device_info: DeviceInfo = DeviceInfo()
        self.current_npu = npu_id
        self.cpu_node: dict[int, int] = {}
        self.numa_to_cpu_map: dict[int, list[int]] = defaultdict(list)
        self.npu_cpu_pool: dict[int, list[int]] = {}
        self.assign_main: dict[int, list[int]] = {}
        self.assign_acl: dict[int, list[int]] = {}
        self.assign_rel: dict[int, list[int]] = {}

    @staticmethod
    def cpu_to_mask(cpu: int) -> str:
        group = cpu // MASK_BIT
        bit = cpu % MASK_BIT
        value = 1 << bit
        mask = f"{value:08x}"
        for _ in range(1, group + 1):
            mask = f"{mask},{'0' * (MASK_BIT // 4)}"
        return mask

    @staticmethod
    def get_threads_map(thread_message: str) -> dict[str, dict[str, list[str]]]:
        threads_map: dict[str, dict[str, list[str]]] = {}
        for line in thread_message.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            main_pid, sub_pid = parts[0], parts[1]
            if "acl_thread" in line:
                key = "acl_thread"
            elif "release_thread" in line:
                key = "release_thread"
            else:
                continue
            if main_pid not in threads_map:
                threads_map[main_pid] = {"acl_thread": [], "release_thread": []}
            threads_map[main_pid][key].append(sub_pid)
        return threads_map

    @staticmethod
    def bind(pid: str, cpus: list[int], bind_sub_thread: bool) -> None:
        if cpus:
            cpu_list = ",".join(map(str, cpus))
            if bind_sub_thread:
                bind_result, return_code = execute_command(["taskset", "-acp", cpu_list, pid])
            else:
                bind_result, return_code = execute_command(["taskset", "-cp", cpu_list, pid])
            if return_code != 0:
                raise RuntimeError(f"Failed to bind {pid} to CPU {cpu_list}.")

    def average_distribute(self, groups: dict[str, list[int]]) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for key, npu_list in groups.items():
            cpu_list = sorted(self.npu_cpu_pool[npu_list[0]])
            cpu_num_per_npu = len(cpu_list) // len(npu_list)
            for i, npu in enumerate(npu_list):
                start_index = i * cpu_num_per_npu
                end_index = (i + 1) * cpu_num_per_npu if i < len(npu_list) - 1 else len(cpu_list)
                result[npu] = cpu_list[start_index:end_index]
        return result

    def extend_numa(self, cpu_list: list[int]) -> list[int]:
        if not cpu_list:
            return []
        nodes = {self.cpu_node[c] for c in cpu_list}
        if len(nodes) != 1:
            return cpu_list
        node = list(nodes)[0]
        next_node = (node + 1) % len(self.numa_to_cpu_map)
        extended = cpu_list[:]
        for cpu in self.numa_to_cpu_map[next_node]:
            if cpu in self.device_info.allowed_cpus:
                extended.append(cpu)
        return sorted(set(extended))

    def build_cpu_node_map(self) -> None:
        cpu_numa_map, _ = execute_command(["lscpu", "-e=CPU,NODE"])
        for line in cpu_numa_map.splitlines():
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            cpu_str, node_str = line.split()
            cpu = int(cpu_str)
            node = int(node_str)
            self.cpu_node[cpu] = node
            self.numa_to_cpu_map[node].append(cpu)
        if len(self.numa_to_cpu_map) == 0:
            raise RuntimeError("lscpu command output error, no NUMA node available. Please check!")

    def build_cpu_pools(self) -> bool:
        self.build_cpu_node_map()
        self._build_topo_affinity_cpu_pool()
        return True

    def _build_fallback_cpu_pool(self) -> None:
        # A2/310P still need a fallback when npu-smi has no affinity topology.
        cpus = sorted(set(self.device_info.allowed_cpus))
        total_npus = (
            self.device_info.total_logic_npus
            or len(self.device_info.npu_affinity)
            or len(self.device_info.running_npu_list)
        )
        if not cpus or not total_npus:
            return
        base, extra = divmod(len(cpus), total_npus)
        if base < MIN_CPUS_PER_NPU:
            raise RuntimeError(f"Insufficient CPUs: each NPU needs at least {MIN_CPUS_PER_NPU} CPUs.")
        for npu in self.device_info.running_npu_list:
            if not 0 <= npu < total_npus:
                raise RuntimeError(f"Invalid NPU id {npu}, total_npus={total_npus}.")
            start = npu * base + min(npu, extra)
            self.npu_cpu_pool[npu] = cpus[start : start + base + (npu < extra)]

    def _build_topo_affinity_cpu_pool(self) -> None:
        # topo_affinity mode
        if not self.device_info.npu_affinity:
            logger.warning("NPU topo affinity not found. action: fallback to global-slice CPU binding.")
            self._build_fallback_cpu_pool()
            return

        allowed_cpu_set = set(self.device_info.allowed_cpus)
        # Consider all logical NPUs to avoid CPU distribution overlap across worker processes.
        candidate_npus = [
            npu
            for npu in self.device_info.all_logic_npus
            # Always keep visible NPUs for strict conflict checks; include
            # non-visible NPUs only when they overlap this process's cpuset.
            if npu in self.device_info.running_npu_list
            or any(cpu in allowed_cpu_set for cpu in self.device_info.npu_affinity.get(npu, []))
        ]
        for npu in candidate_npus:
            base_cpu_list = [
                cpu for cpu in self.device_info.npu_affinity.get(npu, []) if cpu in self.device_info.allowed_cpus
            ]
            if not base_cpu_list:
                raise RuntimeError("CPUs available in 'Cpus_allowed_list' conflict with NUMA affinity.")
            extra_cpu_list = self.extend_numa(base_cpu_list)
            self.npu_cpu_pool[npu] = extra_cpu_list

        groups = defaultdict(list)
        for npu, cpus in self.npu_cpu_pool.items():
            groups[str(cpus)].append(npu)

        final: dict[int, list[int]] = {}
        for key, npu_list in groups.items():
            if len(npu_list) == 1:
                final[npu_list[0]] = self.npu_cpu_pool[npu_list[0]]
            else:
                final.update(self.average_distribute({key: npu_list}))
        # Keep only visible NPUs in the final binding pool. Non-visible NPUs are used only to avoid overlap.
        self.npu_cpu_pool = {npu: final[npu] for npu in self.device_info.running_npu_list}

    def allocate(self) -> None:
        self._allocate_default_roles()

    def _allocate_default_roles(self) -> None:
        for npu, cpu_pool in self.npu_cpu_pool.items():
            if len(cpu_pool) >= MIN_CPUS_PER_NPU:
                main = cpu_pool[2:-2]
                acl = [cpu_pool[-2]]
                rel = [cpu_pool[-1]]
            else:
                raise RuntimeError(
                    f"The number of CPUs is insufficient. Each NPU requires at least {MIN_CPUS_PER_NPU} CPUs."
                )
            self.assign_main[npu] = main
            self.assign_acl[npu] = acl
            self.assign_rel[npu] = rel

    def print_plan(self) -> None:
        logger.info("The CPU allocation plan is as follows:")
        self._print_default_plan(self.current_npu)

    def _print_default_plan(self, current_npu: int) -> None:
        main = " ".join(map(str, self.assign_main[current_npu]))
        acl = " ".join(map(str, self.assign_acl[current_npu]))
        rel = str(self.assign_rel[current_npu]) if self.assign_rel[current_npu] else ""
        logger.info("NPU%s: main=[%s]  acl=[%s]  release=[%s]", current_npu, main, acl, rel)

    def bind_memory(self, pid: str, npu: int) -> None:
        def _get_npu_numa_node(npu_id: int) -> int | None:
            cpu_pool = self.npu_cpu_pool.get(npu_id, [])
            if not cpu_pool:
                return None
            anchor_cpu = cpu_pool[0]
            return self.cpu_node.get(anchor_cpu)

        if not shutil.which("migratepages"):
            logger.info("The 'migratepages' command is not available, skipping memory binding.")
            return
        target_numa = _get_npu_numa_node(npu)
        if target_numa is None:
            logger.warning(
                "[migrate] NPU has no CPU pool. rank=%s, npu=%s.",
                self.rank_id,
                npu,
            )
            return
        all_numa_nodes = sorted(self.numa_to_cpu_map.keys())
        if target_numa not in all_numa_nodes:
            logger.warning("[migrate] NUMA node not found. npu=%s, numa=%s. ", npu, target_numa)
            return
        # Bind memory to the NPU's NUMA node only to minimize cross-NUMA traffic.
        logger.info("[migrate] NPU:%s -> NUMA [%s]", npu, target_numa)
        execute_command(
            [
                "migratepages",
                pid,
                ",".join(map(str, all_numa_nodes)),
                str(target_numa),
            ]
        )

    def bind_threads(self) -> None:
        self._bind_default_threads()

    def _bind_default_threads(self) -> None:
        thread_message, _ = execute_command(["ps", "-Te"])
        threads_map = self.get_threads_map(thread_message)
        main_pid = str(psutil.Process().pid)
        self.bind(main_pid, self.assign_main[self.current_npu], True)
        for thread_id in threads_map.get(main_pid, {}).get("acl_thread", []):
            self.bind(thread_id, self.assign_acl[self.current_npu], False)
        for thread_id in threads_map.get(main_pid, {}).get("release_thread", []):
            self.bind(thread_id, self.assign_rel[self.current_npu], False)

    def bind_npu_irq(self) -> None:
        if not os.access("/proc/irq", os.W_OK):
            logger.warning_once(
                "/proc/irq is not writable. If running in Docker, start the container with '--privileged=true'."
            )
            return

        # Only bind IRQ for current rank's NPU to avoid multi-process overwrite.
        if self.current_npu not in self.npu_cpu_pool:
            logger.warning("[irq] NPU has no CPU pool. rank=%s, npu=%s. ", self.rank_id, self.current_npu)
            return

        if shutil.which("systemctl"):
            output, _ = execute_command(["systemctl", "list-unit-files"])
            if "irqbalance.service" in output:
                _, return_code = execute_command(["systemctl", "is-active", "--quiet", "irqbalance"])
                if return_code == 0:
                    logger.warning(
                        "The irqbalance service is running and has been stopped. "
                        "action: service stopped to enable IRQ binding. "
                    )
                    execute_command(["systemctl", "stop", "irqbalance"])

        sq_irqs = []
        with open("/proc/interrupts") as f:
            for line in f:
                if "sq_send_trigger_irq" in line:
                    irq = line.split(":")[0].strip()
                    sq_irqs.append(irq)

        npu = self.current_npu
        cpus = self.npu_cpu_pool[npu]
        if len(cpus) < 2:
            logger.warning("[irq] CPU pool too small. npu=%s, cpu_count=%d, min_required=2. ", npu, len(cpus))
            return

        sq_cpu, cq_cpu = cpus[0], cpus[1]  # Reserved for IRQ binding
        pci_addr = ""

        info, _ = execute_command(["npu-smi", "info", "-t", "board", "-i", str(npu)])

        for line in info.splitlines():
            if "PCIe Bus Info" in line:
                pci_addr = line.split()[-1].lower()
                break

        if not pci_addr:
            logger.warning("Can't find PCI address. npu=%s. ", npu)
            return

        try:
            npu_irq_list = sorted(os.listdir(f"/sys/bus/pci/devices/{pci_addr}/msi_irqs/"), key=lambda x: int(x))
        except FileNotFoundError:
            logger.warning("The msi_irqs folder cannot be found. pci_addr=%s. ", pci_addr)
            return

        sq_irq, cq_irq = "", ""
        for irq in sq_irqs:
            if irq in npu_irq_list:
                sq_irq = irq
                cq_irq = str(int(irq) + 1)
                break
        if not sq_irq:
            logger.warning("The sq_send_trigger_irq is not found. npu=%s. ", npu)
            return

        logger.info(
            "NPU%s(PCI %s): sq_send_trigger_irq IRQ_ID=%s -> CPU%s, cq_update_irq IRQ_ID=%s -> CPU%s",
            npu,
            pci_addr,
            sq_irq,
            sq_cpu,
            cq_irq,
            cq_cpu,
        )
        with open(f"/proc/irq/{sq_irq}/smp_affinity", "w") as f:
            f.write(self.cpu_to_mask(sq_cpu))
        with open(f"/proc/irq/{cq_irq}/smp_affinity", "w") as f:
            f.write(self.cpu_to_mask(cq_cpu))

    def run_all(self, *, migrate_memory: bool = True) -> None:
        if not self.build_cpu_pools():
            return
        self.allocate()
        self.print_plan()
        self.bind_threads()
        if migrate_memory:
            self.bind_memory(str(psutil.Process().pid), self.current_npu)
        else:
            logger.info("CPU affinity applied without process-wide NUMA page migration.")
        self.bind_npu_irq()


def bind_cpus(
    rank_id: int,
    npu_id: int,
    *,
    migrate_memory: bool = True,
    vllm_config: VllmConfig | None = None,
) -> None:
    if vllm_config is not None:
        from vllm_ascend.ascend_config import validate_additional_config_bool

        binding_config = (vllm_config.additional_config or {}).get("enable_cpu_binding", True)
        if not isinstance(binding_config, dict) and not validate_additional_config_bool(
            binding_config, "enable_cpu_binding"
        ):
            return
    if not is_arm_cpu():
        logger.info("CPU binding skipped: non-ARM CPU detected.")
        return
    if get_ascend_device_type() in (AscendDeviceType.A3, AscendDeviceType.A5):
        _bind_msboost(vllm_config, npu_id, migrate_memory)
        return
    binder = CpuAlloc(rank_id, npu_id=npu_id)
    binder.run_all(migrate_memory=migrate_memory)
