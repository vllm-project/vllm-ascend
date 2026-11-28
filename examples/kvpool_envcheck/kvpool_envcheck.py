#!/usr/bin/env python3
"""Offline vLLM-Ascend kvpool pre-deployment checker, catalog schema 7.1."""

from __future__ import annotations

import argparse
import copy
import datetime
import importlib.metadata
import itertools
import json
import os
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "rules.json"
TOOL_VERSION = "1.1.0"
RESULT_SCHEMA_VERSION = "13.1"
REFERENCE_DOCUMENTS = [
    {
        "title": "vLLM-Ascend KV Cache Pool User Guide",
        "url": "https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md",
        "source_commit": "1985de9f4fd77f9c4a0efdbc98cffc3f43cef305",
        "reviewed_at": "2026-10-20",
    }
]


class CatalogError(ValueError):
    pass


class InputError(ValueError):
    pass


class ProgressReporter:
    def __init__(self, enabled: bool, stream: Any = None):
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr

    @classmethod
    def from_mode(cls, mode: str, stream: Any = None) -> ProgressReporter:
        target = stream if stream is not None else sys.stderr
        enabled = mode == "always" or (mode == "auto" and bool(getattr(target, "isatty", lambda: False)()))
        return cls(enabled, target)

    def _emit(self, message: str) -> None:
        if not self.enabled:
            return
        timestamp = datetime.datetime.now().astimezone().strftime("%H:%M:%S")
        self.stream.write(f"[{timestamp}] {message}\n")
        self.stream.flush()

    @contextmanager
    def stage(self, label: str, index: int | None = None, total: int | None = None):
        prefix = f"[progress {index}/{total}]" if index is not None and total is not None else "[progress]"
        started = time.monotonic()
        self._emit(f"{prefix} {label}: started")
        try:
            yield self
        except BaseException as exc:
            elapsed = time.monotonic() - started
            state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            self._emit(f"{prefix} {label}: {state}, elapsed {elapsed:.1f}s")
            raise
        else:
            elapsed = time.monotonic() - started
            self._emit(f"{prefix} {label}: finished, elapsed {elapsed:.1f}s")


@dataclass(frozen=True)
class CheckContext:
    hardware: str
    backend: str
    managed_memory_mb: int
    deployment_mode: str = "multi-node"
    link_type: str = "roce"
    ssd_offload: bool = False
    ssd_path: str | None = None
    peer_eid: str | None = None
    progress: ProgressReporter | None = None


def run_cmd(argv: list[str], timeout: int = 8) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)


def read_text(path: str, limit: int = 20000) -> str:
    try:
        return Path(path).read_text(errors="replace")[:limit]
    except OSError as exc:
        return f"<unavailable: {exc}>"


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_utf8(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def redact(text: str) -> str:
    text = re.sub(r"(?i)(password|token|passwd)=\S+", r"\1=<redacted>", text)
    text = re.sub(r"(?im)^(\s*(?:serial number|serial|sn)\s*:\s*).*$", r"\1<redacted>", text)
    return text


def compact_sources(sources: list[dict[str, Any]], limit: int = 280) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for item in sources:
        output = redact(item.get("output", ""))
        source = item["source"]
        if source == "npu-smi info":
            topology = parse_npu_topology(output)
            version = re.search(r"(?i)npu-smi\s+([^\s]+)", output)
            summary = (
                f"npu_version={version.group(1) if version else 'UNKNOWN'}; "
                f"topology={json.dumps(topology, ensure_ascii=False, separators=(',', ':'))}"
            )
        else:
            clean_lines = [line for line in output.splitlines() if not re.match(r"^\s*[+\-=|]{5,}\s*$", line)]
            summary = "\n".join(clean_lines)
        summary = summary[:limit]
        key = (item["rc"], summary)
        if key not in grouped:
            grouped[key] = {"source": item["source"], "rc": item["rc"], "summary": summary, "occurrences": 1}
        else:
            grouped[key]["occurrences"] += 1
    return list(grouped.values())


def version_tuple(value: str) -> tuple[int, ...] | None:
    match = re.search(r"(?<!\d)(\d+)(?:\.(\d+))?(?:\.(\d+))?", value or "")
    return tuple(int(item or 0) for item in match.groups()) if match else None


def version_text(value: str) -> str:
    parsed = version_tuple(value)
    return ".".join(str(item) for item in parsed) if parsed else "UNKNOWN"


def run_memory_scan(script: Path, min_segment_mb: int) -> dict[str, Any]:
    """Run mem_scan.py and extract global capacity and diagnostics."""
    rc, out, err = run_cmd([sys.executable, str(script), "-m", str(min_segment_mb)], timeout=600)
    matches = re.findall(r"Total free\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*GB", out)
    total_mb = float(matches[-1]) * 1024 if matches else None
    numa_summary = [
        {"node": int(node), "available_mb": float(free_gb) * 1024, "elapsed": elapsed.strip()}
        for node, free_gb, elapsed in re.findall(r"NUMA node\s+(\d+):\s+([0-9]+(?:\.[0-9]+)?)\s+GB\s+\(([^)]+)\)", out)
    ]
    total_line = re.search(r"Total free\s*:\s*[0-9]+(?:\.[0-9]+)?\s*GB\s+\(([^)]+)\)", out)
    diagnostic = None
    if rc != 0 or total_mb is None:
        diagnostic_lines = [
            line
            for line in (out + "\n" + err).splitlines()
            if line.strip() and "progress" not in line.lower() and not re.match(r"^\s*[-=]{5,}\s*$", line)
        ]
        diagnostic = redact("\n".join(diagnostic_lines[-12:]))
    return {
        "min_segment_mb": min_segment_mb,
        "rc": rc,
        "total_available_mb": total_mb,
        "numa_summary": numa_summary,
        "elapsed": total_line.group(1).strip() if total_line else None,
        "diagnostic": diagnostic,
    }


def probe_a3_memory_scan(managed_memory_mb: int) -> dict[str, Any]:
    """Check A3 global managed-memory capacity, using 1GB then 2MB granularity."""
    script_name = "mem_scan.py"
    script = ROOT / "artifacts" / script_name
    base = {
        "method": "a3_mem_scan",
        "ok": False,
        "script": script_name,
        "scan_scope": "a3_fixed_ranges",
        "managed_memory_mb": managed_memory_mb,
    }
    if not script.exists():
        return {
            **base,
            "failure_kind": "capability",
            "available_memory_mb": None,
            "matched_granularity_mb": None,
            "error": "A3 memory scan script is missing",
            "scans": {},
        }

    # Check at 1GB granularity first, then fall back to 2MB when capacity is insufficient but the scan is valid.
    primary = run_memory_scan(script, 1024)
    primary_valid = primary.get("rc") == 0 and primary.get("total_available_mb") is not None
    primary_pass = primary_valid and primary["total_available_mb"] >= managed_memory_mb
    fallback = None
    if not primary_pass and primary_valid:
        fallback = run_memory_scan(script, 2)
    fallback_total = fallback.get("total_available_mb") if fallback is not None and fallback.get("rc") == 0 else None
    fallback_valid = fallback_total is not None
    fallback_pass = fallback_total is not None and fallback_total >= managed_memory_mb
    ok = bool(primary_pass or fallback_pass)
    valid_scan = primary_valid and (primary_pass or fallback_valid)
    # Compare global available capacity instead of distributing it across NUMA nodes or NPU dies.
    valid_totals = [
        scan["total_available_mb"]
        for scan in (primary, fallback)
        if scan is not None and scan.get("rc") == 0 and scan.get("total_available_mb") is not None
    ]
    available_memory_mb = max(valid_totals) if valid_totals else None
    matched_granularity_mb = 1024 if primary_pass else (2 if fallback_pass else None)
    scans = {"1024": primary, "2": fallback}
    return {
        **base,
        "ok": ok,
        "failure_kind": None if ok else ("capacity" if valid_scan else "capability"),
        "available_memory_mb": available_memory_mb,
        "matched_granularity_mb": matched_granularity_mb,
        "scans": scans,
        "error": None if valid_scan else "No usable scan result was obtained for global capacity evaluation",
    }


def read_proc_meminfo() -> dict[str, str]:
    """Read static huge-page fields from /proc/meminfo."""
    values: dict[str, str] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii", errors="replace").splitlines():
            match = re.match(
                r"^(MemTotal|MemAvailable|HugePages_Total|HugePages_Free|HugePages_Rsvd|Hugepagesize):"
                r"\s+(\d+)\s*(\w+)?$",
                line,
            )
            if match:
                values[match.group(1)] = f"{match.group(2)} {match.group(3) or ''}".strip()
    except OSError:
        return {}
    return values


def parse_meminfo_number(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"^(\d+)", value)
    return int(match.group(1)) if match else None


def probe_a2_static_hugepages(managed_memory_mb: int) -> dict[str, Any]:
    """Check A2 global managed-memory capacity using free 2MB static huge pages."""
    meminfo = read_proc_meminfo()
    page_size_kb = parse_meminfo_number(meminfo.get("Hugepagesize"))
    total_pages = parse_meminfo_number(meminfo.get("HugePages_Total"))
    free_pages = parse_meminfo_number(meminfo.get("HugePages_Free"))
    reserved_pages = parse_meminfo_number(meminfo.get("HugePages_Rsvd"))
    required_pages = (managed_memory_mb + 1) // 2
    evidence_complete = total_pages is not None and free_pages is not None and reserved_pages is not None
    evidence_consistent = False
    if total_pages is not None and free_pages is not None and reserved_pages is not None:
        evidence_consistent = 0 <= reserved_pages <= free_pages <= total_pages
    valid = page_size_kb == 2048 and evidence_consistent
    available_pages = None
    if evidence_consistent and free_pages is not None and reserved_pages is not None:
        available_pages = free_pages - reserved_pages
    available_memory_mb = available_pages * 2 if valid and available_pages is not None else None
    ok = bool(valid and available_pages is not None and available_pages >= required_pages)
    if page_size_kb != 2048:
        error = "Hugepagesize must be 2048kB for the A2 static huge-page check"
    elif not evidence_complete:
        error = "HugePages_Total, HugePages_Free, and HugePages_Rsvd must all be readable"
    elif not evidence_consistent:
        error = "Inconsistent huge-page evidence: Free must not exceed Total and Rsvd must not exceed Free"
    else:
        error = None
    return {
        "method": "a2_static_hugepages",
        "ok": ok,
        "managed_memory_mb": managed_memory_mb,
        "page_size_kb": page_size_kb,
        "required_page_size_kb": 2048,
        "total_pages": total_pages,
        "free_pages": free_pages,
        "reserved_pages": reserved_pages,
        "available_pages": available_pages,
        "required_pages": required_pages,
        "available_memory_mb": available_memory_mb,
        "failure_kind": None if ok else ("capacity" if valid else "capability"),
        "error": error,
    }


def probe_memory_capacity(hardware: str, managed_memory_mb: int) -> dict[str, Any]:
    """Select A2 static huge-page or A3 mem_scan memory checks."""
    if hardware == "A2":
        return probe_a2_static_hugepages(managed_memory_mb)
    if hardware == "A3":
        return probe_a3_memory_scan(managed_memory_mb)
    return {
        "method": "unsupported",
        "ok": False,
        "managed_memory_mb": managed_memory_mb,
        "failure_kind": "capability",
        "available_memory_mb": None,
        "error": f"No memory check strategy is implemented for hardware {hardware}",
    }


def probe_a5_memory_capacity(managed_memory_mb: int) -> dict[str, Any]:
    """Check 950PR&950DT Products host memory availability using MemAvailable."""
    meminfo = read_proc_meminfo()
    total_kb = parse_meminfo_number(meminfo.get("MemTotal"))
    available_kb = parse_meminfo_number(meminfo.get("MemAvailable"))
    total_mb = total_kb / 1024 if total_kb is not None else None
    available_mb = available_kb / 1024 if available_kb is not None else None
    ok = available_mb is not None and available_mb >= managed_memory_mb
    failure_kind: str | None
    if total_mb is None or available_mb is None:
        error = "MemTotal and MemAvailable must both be readable from /proc/meminfo"
        failure_kind = "capability"
    else:
        error = None
        failure_kind = None if ok else "capacity"
    return {
        "method": "a5_host_memory",
        "ok": ok,
        "managed_memory_mb": managed_memory_mb,
        "total_memory_mb": total_mb,
        "available_memory_mb": available_mb,
        "failure_kind": failure_kind,
        "error": error,
    }


def is_docker_runtime() -> bool:
    if Path("/.dockerenv").exists() or Path("/.containerenv").exists():
        return True
    cgroup = read_text("/proc/1/cgroup", 4000).lower()
    return bool(re.search(r"docker|containerd|kubepods|libpod", cgroup))


def probe_docker_prereqs() -> dict[str, Any]:
    """Check generic and 950PR&950DT Products-specific devices, tools, and configuration paths."""
    devices = sorted(str(path) for path in Path("/dev").glob("davinci*"))
    return {
        "docker": is_docker_runtime(),
        "npu_smi": bool(shutil.which("npu-smi")),
        "hccn_tool": bool(find_hccn_tool()),
        "urma_admin": shutil.which("urma_admin") is not None,
        "urma_ping": shutil.which("urma_ping") is not None,
        "devices": devices,
        "davinci_device_count": len(devices),
        "davinci_manager": Path("/dev/davinci_manager").exists(),
        "hisi_hdc": Path("/dev/hisi_hdc").exists(),
        "driver_dir": Path("/usr/local/Ascend/driver").is_dir(),
        "hccn_conf": Path("/etc/hccn.conf").is_file(),
        "ummu_device": Path("/dev/ummu").exists(),
        "uburma_device": Path("/dev/uburma").exists(),
        "hccl_rootinfo": Path("/etc/hccl_rootinfo.json").is_file(),
        "hixlep_dir": Path("/etc/hixlep").is_dir(),
        "route_conf": Path("/lib/route.conf").is_file(),
    }


def find_hccn_tool() -> str | None:
    return shutil.which("hccn_tool") or next(
        (
            str(path)
            for path in (
                Path("/usr/local/Ascend/driver/tools/hccn_tool"),
                Path("/usr/local/Ascend/driver/bin/hccn_tool"),
            )
            if path.is_file() and os.access(path, os.X_OK)
        ),
        None,
    )


def parse_roce_ip(output: str) -> str | None:
    candidates = [
        value
        for value in re.findall(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", output)
        if value not in ("0.0.0.0", "127.0.0.1")
    ]
    return candidates[0] if candidates else None


def parse_roce_netmask(output: str) -> str | None:
    match = re.search(r"(?im)^\s*netmask\s*:\s*((?:\d{1,3}\.){3}\d{1,3})\s*$", output)
    return match.group(1) if match else None


def parse_roce_ping_result(output: str, rc: int, maximum_packet_loss_percent: float = 0.0) -> dict[str, Any]:
    text = output.lower()
    result: dict[str, Any] = {
        "ok": False,
        "failure_kind": "command",
        "reason": "",
        "transmitted": None,
        "received": None,
        "packet_loss_percent": None,
    }
    if "needs the following parameters" in text or "total count of input parameters is wrong" in text:
        result["reason"] = f"hccn_tool reported an invalid parameter (rc={rc})"
        return result
    summary = re.search(
        r"(\d+)\s+packets?\s+transmitted\s*,\s*(\d+)\s+received\s*,\s*([0-9]+(?:\.[0-9]+)?)%\s+packet\s+loss",
        text,
    )
    if summary:
        transmitted, received = int(summary.group(1)), int(summary.group(2))
        packet_loss = float(summary.group(3))
        result.update(transmitted=transmitted, received=received, packet_loss_percent=packet_loss)
        if rc == 0 and transmitted > 0 and packet_loss <= maximum_packet_loss_percent:
            result.update(
                ok=True,
                failure_kind=None,
                reason=(
                    f"Packet loss {packet_loss:.2f}% does not exceed the required maximum "
                    f"{maximum_packet_loss_percent:.2f}%"
                ),
            )
        else:
            result.update(
                failure_kind="connectivity",
                reason=(
                    f"Sent {transmitted} packets, received {received} packets, packet loss {packet_loss:.2f}%, rc={rc}"
                ),
            )
        return result
    if rc != 0:
        result["reason"] = f"hccn_tool command failed (rc={rc})"
        return result
    if re.search(r"timeout|unreachable|failed|failure|error|lost|link\s+down|invalid", text):
        result.update(
            failure_kind="connectivity", reason="Output contains failure, timeout, or unreachable indications"
        )
        return result
    result["reason"] = "The command succeeded but no parseable packet statistics were found"
    return result


ROCE_ERROR_COUNTER_KEYS = (
    "mac_rx_fcs_err_pkt_num",
    "roce_rx_err_pkt_num",
    "roce_tx_err_pkt_num",
    "roce_verification_err_num",
    "roce_qp_status_err_num",
)


def parse_roce_error_counters(output: str) -> dict[str, int]:
    values = {key.strip(): int(value) for key, value in re.findall(r"(?m)^\s*([A-Za-z0-9_]+)\s*:\s*(\d+)\s*$", output)}
    return {key: values.get(key, 0) for key in ROCE_ERROR_COUNTER_KEYS}


def probe_roce_npu_state(tool: str, npu_id: int, parameters: dict[str, Any]) -> dict[str, Any]:
    """Read static RoCE IP, Link, health, and MTU state for one NPU."""
    commands = {
        "ip": [tool, "-i", str(npu_id), "-ip", "-g"],
        "link": [tool, "-i", str(npu_id), "-link", "-g"],
        "health": [tool, "-i", str(npu_id), "-net_health", "-g"],
        "mtu": [tool, "-i", str(npu_id), "-mtu", "-g"],
    }
    outputs = {name: run_cmd(argv, timeout=10) for name, argv in commands.items()}
    ip_text = outputs["ip"][1]
    link_match = re.search(r"(?im)^\s*link\s+status\s*:\s*(\S+)", outputs["link"][1])
    health_match = re.search(r"(?im)^\s*net\s+health\s+status\s*:\s*(.+?)\s*$", outputs["health"][1])
    mtu_match = re.search(r"(?im)^\s*mtu\s*:\s*(\d+)\s*$", outputs["mtu"][1])
    state: dict[str, Any] = {
        "npu_id": npu_id,
        "ip": parse_roce_ip(ip_text),
        "netmask": parse_roce_netmask(ip_text),
        "link_status": link_match.group(1).upper() if link_match else None,
        "net_health": health_match.group(1).strip() if health_match else None,
        "mtu": int(mtu_match.group(1)) if mtu_match else None,
        "command_rc": {name: item[0] for name, item in outputs.items()},
    }
    failures = []
    if any(rc != 0 for rc in state["command_rc"].values()):
        failures.append("At least one hccn_tool query command failed")
    if not state["ip"] or not state["netmask"]:
        failures.append("RoCE IP address or netmask is missing")
    required_link = str(parameters["required_link"]).upper()
    required_health = str(parameters["required_health"])
    if state["link_status"] != required_link:
        failures.append(f"Link={state['link_status'] or 'UNKNOWN'}")
    if (state["net_health"] or "").casefold() != required_health.casefold():
        failures.append(f"net_health={state['net_health'] or 'UNKNOWN'}")
    if not state["mtu"] or state["mtu"] <= 32:
        failures.append(f"MTU={state['mtu']}")
    state["ok"] = not failures
    state["failures"] = failures
    if failures:
        state["diagnostic"] = redact(
            "\n".join((out + "\n" + err).strip() for rc, out, err in outputs.values() if rc != 0 or not out)[-1000:]
        )
    return state


def read_roce_error_counters(tool: str, npu_id: int) -> dict[str, Any]:
    rc, out, err = run_cmd([tool, "-i", str(npu_id), "-stat", "-g"], timeout=10)
    return {
        "rc": rc,
        "counters": parse_roce_error_counters(out) if rc == 0 else {},
        "diagnostic": redact(err[-500:]) if rc != 0 else None,
    }


def run_roce_ping(
    tool: str,
    source: int,
    target: int,
    target_ip: str,
    packet_size: int,
    timeout_seconds: int = 15,
    maximum_packet_loss_percent: float = 0.0,
) -> dict[str, Any]:
    argv = [tool, "-i", str(source), "-ping", "-g", "address", target_ip, "pkt", str(packet_size)]
    rc, out, err = run_cmd(argv, timeout=timeout_seconds)
    return {
        "source_npu": source,
        "target_npu": target,
        "target_ip": target_ip,
        "packet_size": packet_size,
        "command": ["hccn_tool", "-i", str(source), "-ping", "-g", "address", target_ip, "pkt", str(packet_size)],
        "rc": rc,
        "output": redact(out[-1000:]),
        "stderr": redact(err[-500:]),
        **parse_roce_ping_result(out + "\n" + err, rc, maximum_packet_loss_percent),
    }


def probe_roce_diagnostics(
    hardware: str, backend: str, link_parameters: dict[str, Any], local_parameters: dict[str, Any]
) -> dict[str, Any]:
    """Run static RoCE port checks and dynamic local NPU-to-NPU connectivity checks."""
    result: dict[str, Any] = {"hardware": hardware, "backend": backend, "npu_states": [], "link": {}, "local": {}}
    tool = find_hccn_tool()
    if not tool:
        error = "hccn_tool was not found; refusing to fall back to a regular host ping"
        result["link"] = {"ok": False, "failure_kind": "capability", "error": error}
        result["local"] = {
            "ok": False,
            "executed": False,
            "failure_kind": "static_check_failed",
            "error": f"Static port-check capability is unavailable: {error}",
            "directions": [],
        }
        return result
    rc, out, err = run_cmd(["npu-smi", "info"], timeout=10)
    topology = parse_npu_topology(out) if rc == 0 else {}
    npu_ids = topology.get("npu_ids", [])
    result["available_npu_ids"] = npu_ids
    if len(npu_ids) < 2:
        error = f"Fewer than two NPUs are available: {npu_ids}; {redact(err)}"
        result["link"] = {"ok": False, "failure_kind": "capability", "error": error}
        result["local"] = {
            "ok": False,
            "executed": False,
            "failure_kind": "static_check_failed",
            "error": f"Static port check failed: {error}",
            "directions": [],
        }
        return result

    # Check every visible NPU port first; do not run dynamic pair tests after a static failure.
    states = [probe_roce_npu_state(tool, npu_id, link_parameters) for npu_id in npu_ids]
    result["npu_states"] = states
    baseline_failures = [
        f"NPU {state['npu_id']}: {', '.join(state['failures'])}" for state in states if not state["ok"]
    ]
    ips = [state["ip"] for state in states if state["ip"]]
    netmasks = {state["netmask"] for state in states if state["netmask"]}
    mtus = {state["mtu"] for state in states if state["mtu"]}
    if link_parameters["require_unique_ip"] and len(ips) != len(set(ips)):
        baseline_failures.append("Duplicate RoCE IP addresses found")
    if link_parameters["require_consistent_netmask_mtu"]:
        if len(netmasks) != 1:
            baseline_failures.append(f"Inconsistent RoCE netmasks: {sorted(netmasks)}")
        if len(mtus) != 1:
            baseline_failures.append(f"Inconsistent RoCE MTUs: {sorted(mtus)}")
    directions: list[dict[str, Any]] = []
    packet_size = min(mtus) - 32 if len(mtus) == 1 else None
    state_by_id = {state["npu_id"]: state for state in states}
    connectivity_ready = not baseline_failures and packet_size is not None
    before: dict[str, dict[str, Any]] = {}
    after: dict[str, dict[str, Any]] = {}
    # After static ports pass, test all directed pairs and compare error-counter deltas from the test window.
    if connectivity_ready:
        before = {str(npu_id): read_roce_error_counters(tool, npu_id) for npu_id in npu_ids}
        timeout_seconds = int(local_parameters["timeout_seconds"])
        maximum_packet_loss = float(local_parameters["required_packet_loss_percent"])
        pairs = list(itertools.permutations(npu_ids, 2))
        for source, target in pairs:
            directions.append(
                run_roce_ping(
                    tool, source, target, state_by_id[target]["ip"], packet_size, timeout_seconds, maximum_packet_loss
                )
            )
        after = {str(npu_id): read_roce_error_counters(tool, npu_id) for npu_id in npu_ids}

    counter_deltas: dict[str, dict[str, int]] = {}
    counter_failures = []
    counter_read_failure = False
    for npu_id in npu_ids if connectivity_ready else []:
        key = str(npu_id)
        if before[key]["rc"] != 0 or after[key]["rc"] != 0:
            counter_read_failure = True
            counter_failures.append(f"Failed to read error counters for NPU {npu_id}")
            continue
        counter_deltas[key] = {}
        for counter in ROCE_ERROR_COUNTER_KEYS:
            old = before[key]["counters"].get(counter, 0)
            new = after[key]["counters"].get(counter, 0)
            delta = new - old if new >= old else new
            counter_deltas[key][counter] = delta
            if delta > 0:
                counter_failures.append(f"NPU {npu_id} {counter} increased by {delta}")
    error_counters = {"before": before, "after": after, "deltas": counter_deltas}

    result["link"] = {
        "ok": not baseline_failures,
        "failure_kind": None if not baseline_failures else "port_readiness",
        "error": "; ".join(baseline_failures) if baseline_failures else None,
    }
    failed_directions = [item for item in directions if not item["ok"]]
    if not connectivity_ready:
        result["local"] = {
            "ok": False,
            "executed": False,
            "failure_kind": "static_check_failed",
            "packet_size": packet_size,
            "direction_count": 0,
            "directions": directions,
            "error": "Static port-readiness checks failed; local NPU-to-NPU data-path tests were not run",
        }
    else:
        local_failures = [
            item.get("reason", "Directed NPU-pair test failed") for item in failed_directions
        ] + counter_failures
        result["local"] = {
            "ok": not local_failures,
            "executed": True,
            "failure_kind": None
            if not local_failures
            else (
                "capability"
                if counter_read_failure
                else (
                    "command"
                    if any(item.get("failure_kind") == "command" for item in failed_directions)
                    else "data_path"
                )
            ),
            "packet_size": packet_size,
            "direction_count": len(directions),
            "directions": directions,
            "error_counters": error_counters,
            "error": "; ".join(
                (
                    [f"{len(failed_directions)}/{len(directions)} directed NPU-pair tests failed"]
                    if failed_directions
                    else []
                )
                + counter_failures
            )
            or None,
        }
    return result


def parse_hccs_state(output: str) -> dict[str, Any]:
    def parse_values(label: str, numeric: bool = True) -> list[Any]:
        match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*\[([^]]*)\]", output)
        if not match:
            return []
        values = match.group(1).split()
        if not numeric:
            return values
        try:
            return [int(value) for value in values]
        except ValueError:
            return []

    health = re.search(r"(?im)^\s*hccs health status\s*:\s*(\S+)", output)
    first_error = re.search(r"(?im)^\s*hccs first error lane\s*:\s*(\S+)", output)
    return {
        "health_status": health.group(1).upper() if health else None,
        "lane_mode": parse_values("hccs lane mode"),
        "link_lane_list": parse_values("hccs link lane list", numeric=False),
        "link_speed": parse_values("hccs link speed"),
        "retry_count": parse_values("hccs retry count"),
        "error_count": parse_values("hccs error count"),
        "first_error_lane": first_error.group(1) if first_error else None,
    }


def parse_hccs_topology(output: str) -> dict[int, dict[int, str]]:
    header: list[int] = []
    rows: dict[int, dict[int, str]] = {}
    for line in output.splitlines():
        phy_ids = [int(value) for value in re.findall(r"Phy-ID(\d+)", line)]
        if len(phy_ids) > 1 and not header:
            header = phy_ids
            continue
        match = re.match(r"^\s*Phy-ID(\d+)\s+(.+?)\s*$", line)
        if not match or not header:
            continue
        values = match.group(2).split()
        if len(values) >= len(header):
            rows[int(match.group(1))] = dict(zip(header, values[: len(header)]))
    return rows


def parse_npu_processes(output: str) -> list[dict[str, Any]]:
    header = re.search(r"(?im)^\|\s*NPU\s+Chip\s*\|\s*Process id", output)
    if not header:
        return []
    processes = []
    for line in output[header.end() :].splitlines():
        columns = [column.strip() for column in line.split("|")]
        if len(columns) < 4:
            continue
        location = re.fullmatch(r"(\d+)\s+(\d+)", columns[1])
        pid = re.fullmatch(r"\d+", columns[2])
        if not location or not pid:
            continue
        processes.append(
            {
                "npu_id": int(location.group(1)),
                "chip_id": int(location.group(2)),
                "pid": int(pid.group(0)),
                "process_name": columns[3][:160],
            }
        )
    return processes


def query_hccs_states(
    device_pairs: list[tuple[int, int]], parameters: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str], bool]:
    states = []
    failures: list[str] = []
    capability_failure = False
    for npu_id, chip_id in device_pairs:
        rc, out, err = run_cmd(["npu-smi", "info", "-t", "hccs", "-i", str(npu_id), "-c", str(chip_id)], timeout=15)
        parsed = parse_hccs_state(out) if rc == 0 else {}
        state = {"npu_id": npu_id, "chip_id": chip_id, "rc": rc, **parsed}
        state_failures = []
        if rc != 0:
            capability_failure = True
            state_failures.append(f"Query failed, rc={rc}: {redact(err[-300:])}")
        elif state.get("health_status") != str(parameters["required_health"]).upper():
            state_failures.append(f"health={state.get('health_status') or 'UNKNOWN'}")
        lengths = [
            len(state.get(key, []))
            for key in ("lane_mode", "link_lane_list", "link_speed", "retry_count", "error_count")
        ]
        if rc == 0 and (not lengths[0] or len(set(lengths)) != 1):
            capability_failure = True
            state_failures.append(f"HCCS link fields are missing or have inconsistent lengths: {lengths}")
        if rc == 0 and lengths[0] and len(set(lengths)) == 1:
            for index, (lane_mode, lane_list, speed) in enumerate(
                zip(state["lane_mode"], state["link_lane_list"], state["link_speed"]), start=1
            ):
                if parameters["require_complete_lanes"] and (
                    lane_mode <= 0 or not re.fullmatch(r"1+", lane_list) or len(lane_list) != lane_mode
                ):
                    state_failures.append(
                        f"link {index} has an invalid lane configuration: mode={lane_mode}, list={lane_list}"
                    )
                if parameters["require_positive_speed"] and speed <= 0:
                    state_failures.append(f"link {index} speed={speed}")
        state["ok"] = not state_failures
        state["failures"] = state_failures
        states.append(state)
        failures.extend(f"card {npu_id}/die {chip_id}: {failure}" for failure in state_failures)
    return states, failures, capability_failure


def hccs_counter_snapshot(states: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    return {
        f"{state['npu_id']}:{state['chip_id']}": {
            "retry_count": list(state.get("retry_count", [])),
            "error_count": list(state.get("error_count", [])),
        }
        for state in states
    }


def hccs_counter_deltas(
    before: dict[str, dict[str, list[int]]], after: dict[str, dict[str, list[int]]], maximum_delta: int = 0
) -> tuple[dict[str, dict[str, list[int]]], list[str]]:
    deltas: dict[str, dict[str, list[int]]] = {}
    failures = []
    for device, old_counters in before.items():
        new_counters = after.get(device)
        if new_counters is None:
            failures.append(f"Counters for card/die {device} are missing after the test")
            continue
        deltas[device] = {}
        for name in ("retry_count", "error_count"):
            old_values = old_counters.get(name, [])
            new_values = new_counters.get(name, [])
            if len(old_values) != len(new_values):
                failures.append(f"card/die {device} {name} length changed")
                continue
            values = [new - old if new >= old else new for old, new in zip(old_values, new_values)]
            deltas[device][name] = values
            for link_index, value in enumerate(values, start=1):
                if value > maximum_delta:
                    failures.append(f"card/die {device} link {link_index} {name} increased by {value}")
    return deltas, failures


def run_hccs_collective(
    script: Path, world_size: int, elements_per_peer: int, iterations: int, timeout_seconds: int
) -> dict[str, Any]:
    """Run an isolated HCCL collective test and parse its result."""
    argv = [
        sys.executable,
        str(script),
        "--world-size",
        str(world_size),
        "--elements-per-peer",
        str(elements_per_peer),
        "--iterations",
        str(iterations),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    started = time.monotonic()
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        out, err = proc.communicate(timeout=timeout_seconds + 30)
        rc = proc.returncode
    except KeyboardInterrupt:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.communicate(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
        raise
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        rc = 124
        err = (err + "\nHCCL dynamic test timed out; the entire test process group was terminated").strip()
    except OSError as exc:
        return {
            "ok": False,
            "rc": 127,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "result": None,
            "diagnostic": str(exc),
        }
    parsed = None
    for line in reversed(out.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    return {
        "ok": bool(rc == 0 and parsed and parsed.get("ok")),
        "rc": rc,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "result": parsed,
        "diagnostic": redact("\n".join(line for line in (out + "\n" + err).splitlines() if line.strip())[-4000:])
        if rc != 0 or not parsed
        else None,
    }


def probe_hccs_diagnostics(
    hardware: str, backend: str, static_parameters: dict[str, Any], dynamic_parameters: dict[str, Any]
) -> dict[str, Any]:
    """Run A3 SIO/HCCS static topology checks and HCCL dynamic data-path tests."""
    result: dict[str, Any] = {"hardware": hardware, "backend": backend, "static": {}, "dynamic": {}}
    rc, npu_output, npu_err = run_cmd(["npu-smi", "info"], timeout=15)
    topology = parse_npu_topology(npu_output) if rc == 0 else {}
    chip_ids_by_npu = topology.get("chip_ids_by_npu", {})
    device_pairs = [
        (npu_id, chip_id)
        for npu_id in topology.get("npu_ids", [])
        for chip_id in chip_ids_by_npu.get(npu_id, list(range(topology.get("chips_by_npu", {}).get(npu_id, 0))))
    ]
    if rc != 0 or not device_pairs:
        error = f"Failed to enumerate A3 cards/dies: npu-smi rc={rc}; {redact(npu_err[-500:])}"
        result["static"] = {"ok": False, "failure_kind": "capability", "device_pairs": device_pairs, "error": error}
        result["dynamic"] = {"ok": False, "executed": False, "failure_kind": "static_check_failed", "error": error}
        return result

    # Check each visible die's HCCS state, then validate static card/die topology relations.
    states, state_failures, capability_failure = query_hccs_states(device_pairs, static_parameters)
    first_npu, first_chip = device_pairs[0]
    topo_rc, topo_out, topo_err = run_cmd(
        ["npu-smi", "info", "-t", "topo", "-i", str(first_npu), "-c", str(first_chip)], timeout=15
    )
    topology_rows = parse_hccs_topology(topo_out) if topo_rc == 0 else {}
    physical_ids_by_npu = topology.get("physical_ids_by_npu", {})
    visible_physical_ids = sorted({value for values in physical_ids_by_npu.values() for value in values})
    physical_to_npu = {physical_id: npu_id for npu_id, values in physical_ids_by_npu.items() for physical_id in values}
    relation_counts: dict[str, int] = {}
    topology_failures = []
    if topo_rc != 0 or not topology_rows or len(visible_physical_ids) != len(device_pairs):
        capability_failure = True
        topology_failures.append(
            f"Topology evidence is incomplete: topo rc={topo_rc}, visible physical IDs={visible_physical_ids}, "
            f"cards/dies={len(device_pairs)}; {redact(topo_err[-300:])}"
        )
    else:
        for source, target in itertools.permutations(visible_physical_ids, 2):
            relation = topology_rows.get(source, {}).get(target)
            relation_counts[relation or "MISSING"] = relation_counts.get(relation or "MISSING", 0) + 1
            expected = (
                {str(static_parameters["same_card_relation"])}
                if physical_to_npu[source] == physical_to_npu[target]
                else {str(value) for value in static_parameters["cross_card_relations"]}
            )
            if relation not in expected:
                topology_failures.append(
                    f"Phy-ID{source}->Phy-ID{target}={relation or 'MISSING'}, expected={'/'.join(sorted(expected))}"
                )
    failures = state_failures + topology_failures
    static_ok = not failures
    before_counters = hccs_counter_snapshot(states)
    result["static"] = {
        "ok": static_ok,
        "failure_kind": None if static_ok else ("capability" if capability_failure else "link"),
        "visible_npu_ids": topology.get("npu_ids", []),
        "device_pairs": [{"npu_id": npu_id, "chip_id": chip_id} for npu_id, chip_id in device_pairs],
        "visible_physical_ids": visible_physical_ids,
        "topology_relation_counts": relation_counts,
        "dies": states,
        "error": "; ".join(failures) if failures else None,
    }
    # Skip dynamic tests after a static topology or link failure to avoid duplicating the same blocker.
    if not static_ok:
        result["dynamic"] = {
            "ok": False,
            "executed": False,
            "failure_kind": "static_check_failed",
            "error": "Static topology or link checks failed; HCCL dynamic communication tests were not run",
        }
        return result

    # Dynamic tests require exclusive access to all visible dies and are skipped when workloads are running.
    processes = parse_npu_processes(npu_output)
    if processes:
        result["dynamic"] = {
            "ok": False,
            "executed": False,
            "failure_kind": "devices_busy",
            "processes": processes,
            "error": (
                f"Detected {len(processes)} NPU processes; exclusive HCCL dynamic tests were not run "
                "to avoid impacting workloads"
            ),
        }
        return result

    script = ROOT / "artifacts" / "hccs_test_a3.py"
    if not script.exists():
        result["dynamic"] = {
            "ok": False,
            "executed": False,
            "failure_kind": "capability",
            "error": "Dynamic test script artifacts/hccs_test_a3.py is missing",
        }
        return result
    # Re-read counters after the dynamic test and evaluate only deltas from this test window.
    collective = run_hccs_collective(
        script,
        len(device_pairs),
        elements_per_peer=int(dynamic_parameters["elements_per_peer"]),
        iterations=int(dynamic_parameters["iterations"]),
        timeout_seconds=int(dynamic_parameters["timeout_seconds"]),
    )
    after_states, after_failures, after_capability_failure = query_hccs_states(device_pairs, static_parameters)
    after_counters = hccs_counter_snapshot(after_states)
    deltas, counter_failures = hccs_counter_deltas(
        before_counters, after_counters, int(dynamic_parameters["required_retry_error_delta"])
    )
    dynamic_failures = (
        ([] if collective.get("ok") else ["HCCL all-to-all communication or data correctness verification failed"])
        + after_failures
        + counter_failures
    )
    result["dynamic"] = {
        "ok": not dynamic_failures,
        "executed": True,
        "failure_kind": None
        if not dynamic_failures
        else ("capability" if after_capability_failure or collective.get("rc") in (124, 127) else "data_path"),
        "world_size": len(device_pairs),
        "collective": collective,
        "error_counters": {"before": before_counters, "after": after_counters, "deltas": deltas},
        "error": "; ".join(dynamic_failures) if dynamic_failures else None,
    }
    return result


def probe_package_versions(names: list[str]) -> dict[str, Any]:
    """Read package versions required by the applicable rules."""
    versions: dict[str, str | None] = {}
    for name in dict.fromkeys(names):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"versions": versions}


def probe_ssd_storage(path_text: str, backend: str) -> dict[str, Any]:
    """Check SSD Offload path type, mount state, and availability for the backend."""
    if not path_text:
        return {
            "path": path_text,
            "backend": backend,
            "ok": False,
            "error": "--ssd-path is required when SSD Offload is enabled",
        }
    path = Path(path_text)
    result: dict[str, Any] = {"path": path_text, "backend": backend, "ok": False}
    if not path.exists():
        result["error"] = "SSD path does not exist"
        return result
    if path.is_dir():
        if backend == "memcache":
            result["error"] = (
                "Memcache SSD requires an unmounted block device, partition, or loop device, not a directory"
            )
            return result
        if backend != "mooncake":
            result["error"] = f"No SSD Offload storage check strategy is implemented for backend {backend}"
            return result
        try:
            stat = os.statvfs(path)
            result["free_bytes"] = stat.f_bavail * stat.f_frsize
            result["ok"] = os.access(path, os.W_OK) and result["free_bytes"] > 0
            if not result["ok"]:
                result["error"] = "SSD directory is not writable or has no free space"
            return result
        except OSError as exc:
            result["error"] = f"Failed to inspect SSD directory: {exc}"
            return result
    if not path.is_block_device():
        result["error"] = "SSD path is neither a directory nor a block device"
        return result
    if backend not in ("memcache", "mooncake"):
        result["error"] = f"No SSD Offload storage check strategy is implemented for backend {backend}"
        return result
    rc, out, err = run_cmd(["lsblk", "-J", "-o", "PATH,TYPE,FSTYPE,MOUNTPOINTS,PKNAME,SIZE", str(path)], timeout=10)
    result.update({"lsblk_rc": rc, "lsblk_error": err})
    if rc != 0:
        result["error"] = "lsblk could not read SSD device information"
        return result
    try:
        data = json.loads(out)
        devices = data.get("blockdevices", [])
        selected = next((item for item in devices if item.get("path") == str(path)), devices[0] if devices else {})
        mounts = selected.get("mountpoints") or []
        fstype = selected.get("fstype")
        selected_name = Path(str(selected.get("path", ""))).name
        partitions = [item for item in devices if item.get("type") == "part" and item.get("pkname") == selected_name]
        result["device"] = selected
        result["partitions"] = partitions
        result["partition_count"] = len(partitions)
        result["ok"] = not fstype and not any(mounts) and not partitions
        if not result["ok"]:
            result["error"] = "SSD device has partitions, a filesystem signature, or a mount point"
    except json.JSONDecodeError as exc:
        result["error"] = f"Failed to parse lsblk output: {exc}"
    return result


def probe_glibc() -> dict[str, Any]:
    """Read glibc version and ldd diagnostics from the container."""
    rc, out, err = run_cmd(["ldd", "--version"], timeout=10)
    match = re.search(r"(?i)glibc[^0-9]*(\d+\.\d+)|ldd[^0-9]*(\d+\.\d+)", out or err)
    return {
        "version": next((item for item in (match.groups() if match else ()) if item), None),
        "rc": rc,
        "summary": (out or err)[:300],
    }


def probe_hdk() -> dict[str, Any]:
    """Read HDK/CANN versions from driver, npu-smi, and Toolkit information visible in the container."""
    sources: list[dict[str, Any]] = []
    for argv in (["npu-smi", "info"], ["npu-smi", "info", "-t", "board", "-i", "0"]):
        rc, out, err = run_cmd(list(argv), timeout=10)
        if out or err:
            sources.append({"source": " ".join(argv), "rc": rc, "output": redact(out or err)})
    paths = [
        "/usr/local/Ascend/driver/version.info",
        "/usr/local/Ascend/driver/version.info.txt",
        "/etc/ascend_install.info",
    ]
    toolkit_roots = [Path("/usr/local/Ascend/ascend-toolkit/latest")]
    for env_name in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        if os.environ.get(env_name):
            toolkit_roots.append(Path(os.environ[env_name]))
    for root in toolkit_roots:
        paths.extend(str(path) for path in root.glob("**/ascend_toolkit_install.info"))
        paths.extend(str(path) for path in root.glob("**/version.cfg"))
    for path in dict.fromkeys(paths):
        if Path(path).exists():
            sources.append({"source": path, "rc": 0, "output": redact(read_text(path, 4000))})
    sources.sort(
        key=lambda item: int(bool(re.search(r"(?i)(hdk|firmware|driver|version)", item["output"]))), reverse=True
    )
    version_pattern = r"\d+\.\d+(?:\.\d+)?(?:[.-]?(?:rc|RC)\d+)?"
    found = None
    found_source = None
    cann_found = None
    cann_source = None
    for item in sources:
        match = re.search(rf"(?i)(?:hdk|firmware|driver|version)[^0-9]{{0,20}}({version_pattern})", item["output"])
        if not match:
            match = re.search(rf"(?<!\d)({version_pattern})(?!\d)", item["output"])
        if match:
            source_evidence = {
                "source": item["source"],
                "rc": item["rc"],
                "matched_line": next(
                    (line.strip() for line in item["output"].splitlines() if match.group(1) in line), ""
                )[:280],
            }
            is_cann = "toolkit" in item["source"].lower() or "cann" in item.get("output", "").lower()
            if is_cann:
                cann_found = cann_found or match.group(1)
                cann_source = cann_source or source_evidence
            else:
                found = found or match.group(1)
                found_source = found_source or source_evidence
    return {"version": found, "version_source": found_source, "cann_version": cann_found, "cann_source": cann_source}


def probe_kernel() -> dict[str, Any]:
    """Read the Linux kernel version visible to the container."""
    rc, out, err = run_cmd(["uname", "-r"])
    return {"value": out if rc == 0 else platform.release(), "rc": rc, "stderr": err}


def parse_npu_topology(output: str) -> dict[str, Any]:
    npu_ids = set()
    chips_by_npu: dict[int, int] = {}
    chip_ids_by_npu: dict[int, list[int]] = {}
    physical_ids_by_npu: dict[int, list[int]] = {}
    names_by_npu: dict[int, set[str]] = {}
    in_device_table = False
    separate_id_name_columns = False
    current_npu: int | None = None
    for line in output.splitlines():
        if re.search(r"\|\s*NPU\s+ID\s*\|\s*Name\s*\|", line, re.IGNORECASE):
            separate_id_name_columns = True
            in_device_table = True
            continue
        if re.search(r"\|\s*NPU\s+Name\s+\|", line, re.IGNORECASE):
            in_device_table = True
            continue
        if in_device_table and re.search(r"\|\s*NPU\s+(?:Chip|ID)\s*\|\s*(?:Process id|Chip ID)", line, re.IGNORECASE):
            break
        if not in_device_table:
            continue
        columns = line.split("|")
        if len(columns) < 3:
            continue
        if separate_id_name_columns:
            id_match = re.fullmatch(r"\s*(\d+)\s*", columns[1])
            device_name = columns[2].strip()
            if not id_match or not device_name:
                continue
            npu_id = int(id_match.group(1))
            npu_ids.add(npu_id)
            current_npu = npu_id
            chips_by_npu[npu_id] = chips_by_npu.get(npu_id, 0) + 1
            names_by_npu.setdefault(npu_id, set()).add(device_name)
            continue
        match = re.match(r"^\s*(\d+)\s+(.+?)\s*$", columns[1])
        if not match:
            continue
        npu_id = int(match.group(1))
        device_name = match.group(2).strip()
        if not device_name:
            continue
        chip_match = re.fullmatch(r"(\d+)(?:\s+(\d+))?", f"{npu_id} {device_name}")
        if chip_match and current_npu is not None:
            chip_id = int(chip_match.group(1))
            chip_ids_by_npu.setdefault(current_npu, []).append(chip_id)
            if chip_match.group(2) is not None:
                physical_ids_by_npu.setdefault(current_npu, []).append(int(chip_match.group(2)))
            continue
        npu_ids.add(npu_id)
        current_npu = npu_id
        chips_by_npu[npu_id] = chips_by_npu.get(npu_id, 0) + 1
        names_by_npu.setdefault(npu_id, set()).add(device_name)
    chip_counts = sorted(set(chips_by_npu.values()))
    device_names = sorted({name for names in names_by_npu.values() for name in names})
    return {
        "npu_count": len(npu_ids),
        "total_chip_count": sum(chips_by_npu.values()),
        "chips_per_npu": chip_counts[0] if len(chip_counts) == 1 else None,
        "npu_ids": sorted(npu_ids),
        "chips_by_npu": chips_by_npu,
        "chip_ids_by_npu": {npu_id: sorted(set(values)) for npu_id, values in sorted(chip_ids_by_npu.items())},
        "physical_ids_by_npu": {npu_id: sorted(set(values)) for npu_id, values in sorted(physical_ids_by_npu.items())},
        "device_names": device_names,
        "names_by_npu": {npu_id: sorted(names) for npu_id, names in sorted(names_by_npu.items())},
    }


def parse_npu_mapping_ids(output: str) -> list[int]:
    """Parse NPU IDs from ``npu-smi info -m`` mapping rows."""
    ids = set()
    for line in output.splitlines():
        match = re.match(r"^\s*(\d+)\s+\d+\s+\d+\s+\d+\s+\S+", line)
        if match:
            ids.add(int(match.group(1)))
    return sorted(ids)


def parse_a5_topology(output: str, expected_relation: str) -> dict[str, Any]:
    """Parse the 950PR&950DT Products NPU topology matrix and validate off-diagonal UB links."""
    npu_ids: list[int] = []
    for line in output.splitlines():
        header_ids = [int(value) for value in re.findall(r"NPU(\d+)", line)]
        if len(header_ids) >= 2:
            npu_ids = header_ids
            break
    rows: dict[int, dict[int, str]] = {}
    if npu_ids:
        for line in output.splitlines():
            fields = line.split()
            if not fields or not re.fullmatch(r"NPU\d+", fields[0]):
                continue
            source = int(fields[0][3:])
            relations = fields[1 : len(npu_ids) + 1]
            if len(relations) != len(npu_ids):
                continue
            rows[source] = dict(zip(npu_ids, relations))
    mismatches = [
        {"source": source, "destination": destination, "relation": relation}
        for source, row in rows.items()
        for destination, relation in row.items()
        if source != destination and relation.upper() != expected_relation.upper()
    ]
    missing_rows = sorted(set(npu_ids) - set(rows))
    diagonal_invalid = sorted(source for source, row in rows.items() if row.get(source, "").upper() != "X")
    ok = bool(npu_ids) and not missing_rows and not mismatches and not diagonal_invalid
    return {
        "ok": ok,
        "npu_ids": npu_ids,
        "rows": {str(source): row for source, row in sorted(rows.items())},
        "missing_rows": missing_rows,
        "mismatches": mismatches,
        "diagonal_invalid": diagonal_invalid,
        "required_relation": expected_relation,
    }


def parse_a5_port_table(output: str) -> dict[str, Any]:
    """Parse 950PR&950DT Products UB port rows from ``hccn_tool -g -dev_info``."""
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if "|" not in line:
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) != 6 or not columns[0].isdigit() or not columns[1].isdigit():
            continue
        speed_match = re.match(r"(\d+)", columns[2])
        if not speed_match:
            continue
        rows.append(
            {
                "udie_id": int(columns[0]),
                "port_id": int(columns[1]),
                "speed_gbps": int(speed_match.group(1)),
                "port_type": columns[3],
                "link_status": columns[4],
                "media_type": columns[5],
            }
        )
    return {"rows": rows, "ok": bool(rows), "error": None if rows else "No UB port rows were parsed"}


def parse_urma_endpoints(output: str) -> list[dict[str, Any]]:
    """Parse active and inactive URMA endpoint rows."""
    endpoints = []
    for line in output.splitlines():
        match = re.match(
            r"^\s*\d+\s+(\S+)\s+(\S+)\s+eid\d+\s+"
            r"([0-9a-fA-F:]+)\s+(ACTIVE|INACTIVE)\s*$",
            line,
        )
        if match:
            endpoints.append(
                {
                    "ubep_dev": match.group(1),
                    "tp_type": match.group(2),
                    "eid": match.group(3),
                    "link": match.group(4),
                }
            )
    return endpoints


def probe_a5_hugepages() -> dict[str, Any]:
    """Read 950PR&950DT Products 1GB and 2MB huge-page allocation on every visible NUMA node."""
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    node_root = Path("/sys/devices/system/node")
    nodes = sorted(
        (path for path in node_root.glob("node[0-9]*") if path.is_dir()),
        key=lambda path: int(path.name[4:]),
    )
    if not nodes:
        errors.append("No NUMA nodes found under /sys/devices/system/node")
    for node in nodes:
        for size_kb in (2048, 1048576):
            path = node / f"hugepages/hugepages-{size_kb}kB/nr_hugepages"
            try:
                value = int(path.read_text(encoding="ascii").strip())
            except (OSError, ValueError) as exc:
                errors.append(f"{path}: {exc}")
                continue
            entries.append(
                {
                    "node": int(node.name[4:]),
                    "page_size_kb": size_kb,
                    "configured_pages": value,
                    "path": str(path),
                }
            )
    residual = [entry for entry in entries if entry["configured_pages"] > 0]
    evidence_complete = not errors and len(entries) == len(nodes) * 2
    return {
        "method": "a5_hugepages",
        "ok": evidence_complete and not residual,
        "evidence_complete": evidence_complete,
        "entries": entries,
        "residual_entries": residual,
        "errors": errors,
    }


def probe_a5_secverify() -> dict[str, Any]:
    """Read the documented 950PR&950DT Products custom-op signature verification state."""
    mapping_rc, mapping_output, mapping_error = run_cmd(["npu-smi", "info", "-m"], timeout=15)
    npu_ids = parse_npu_mapping_ids(mapping_output)
    states = []
    errors = []
    if mapping_rc != 0 or not npu_ids:
        errors.append(
            f"npu-smi info -m rc={mapping_rc}; {redact(mapping_error or mapping_output or 'no NPU mapping output')}"
        )
    for npu_id in npu_ids:
        enable_rc, enable_output, enable_error = run_cmd(
            ["npu-smi", "info", "-t", "custom-op-secverify-enable", "-i", str(npu_id)],
            timeout=15,
        )
        mode_rc, mode_output, mode_error = run_cmd(
            ["npu-smi", "info", "-t", "custom-op-secverify-mode", "-i", str(npu_id)],
            timeout=15,
        )
        enable_match = re.search(r"custom-op-secverify-enable\s*:\s*(True|False)", enable_output, re.IGNORECASE)
        mode_match = re.search(r"custom-op-secverify-mode\s*:\s*(\d+)", mode_output, re.IGNORECASE)
        state = {
            "npu_id": npu_id,
            "enable": enable_match.group(1).lower() == "true" if enable_match else None,
            "mode": int(mode_match.group(1)) if mode_match else None,
            "command_rc": {"enable": enable_rc, "mode": mode_rc},
        }
        states.append(state)
        if enable_rc != 0 or mode_rc != 0 or state["enable"] is None or state["mode"] is None:
            errors.append(
                f"NPU {npu_id}: {redact(enable_error or enable_output or mode_error or mode_output or 'no output')}"
            )
    return {
        "npu_ids": npu_ids,
        "states": states,
        "errors": errors,
        "ok": bool(states) and not errors,
    }


def parse_urma_ping(output: str, rc: int, maximum_packet_loss_percent: float = 0.0) -> dict[str, Any]:
    """Parse ``urma_ping`` loss counters and command status."""
    match = re.search(
        r"(\d+)\s+packets?\s+transmitted\s*,\s*(\d+)\s+received\s*,\s*([0-9]+(?:\.[0-9]+)?)%\s+packet loss",
        output,
        re.IGNORECASE,
    )
    if match:
        transmitted = int(match.group(1))
        received = int(match.group(2))
        packet_loss = float(match.group(3))
        return {
            "ok": rc == 0
            and transmitted > 0
            and received == transmitted
            and packet_loss <= maximum_packet_loss_percent,
            "transmitted": transmitted,
            "received": received,
            "packet_loss_percent": packet_loss,
        }
    return {
        "ok": False,
        "transmitted": None,
        "received": None,
        "packet_loss_percent": None,
    }


def probe_a5_ub(context: CheckContext, rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Probe 950PR&950DT Products NPU topology, UB ports, URMA endpoints, and optional peer ping."""
    static_parameters = evaluator_parameters(rules, "a5_ub_static")
    ping_parameters = evaluator_parameters(rules, "a5_peer_ping")
    mapping_rc, mapping_output, mapping_error = run_cmd(["npu-smi", "info", "-m"], timeout=15)
    npu_ids = parse_npu_mapping_ids(mapping_output)
    topology_rc, topology_output, topology_error = run_cmd(["npu-smi", "info", "-t", "topo"], timeout=20)
    topology = parse_a5_topology(topology_output, str(static_parameters.get("required_topology_relation", "UB")))
    hccn_tool = find_hccn_tool()
    port_results = []
    if hccn_tool and npu_ids:
        for npu_id in npu_ids:
            rc, output, error = run_cmd([hccn_tool, "-g", "-dev_info", "-i", str(npu_id)], timeout=15)
            ports = parse_a5_port_table(output)
            port_results.append(
                {
                    "npu_id": npu_id,
                    "rc": rc,
                    "error": redact(error),
                    **ports,
                }
            )
    static_errors = []
    if mapping_rc != 0 or not npu_ids:
        static_errors.append(
            f"npu-smi info -m rc={mapping_rc}; {redact(mapping_error or mapping_output or 'no NPU mapping output')}"
        )
    if topology_rc != 0:
        static_errors.append(
            f"npu-smi info -t topo rc={topology_rc}; "
            f"{redact(topology_error or topology_output or 'no topology output')}"
        )
    if not hccn_tool:
        static_errors.append("hccn_tool is not available")
    for result in port_results:
        if result["rc"] != 0 or not result["ok"]:
            static_errors.append(
                f"NPU {result['npu_id']} port query rc={result['rc']}; "
                f"{redact(result['error'] or 'no parsable port rows')}"
            )
    for result in port_results:
        for port in result["rows"]:
            if (
                port["port_type"] not in static_parameters.get("allowed_port_types", [])
                or port["link_status"].upper() != str(static_parameters.get("required_link", "UP")).upper()
                or (static_parameters.get("require_positive_speed", True) and port["speed_gbps"] <= 0)
            ):
                static_errors.append(
                    f"NPU {result['npu_id']} UDie {port['udie_id']} port {port['port_id']} "
                    f"is {port['port_type']}/{port['link_status']}/{port['speed_gbps']}Gbps"
                )
    capability_failed = (
        mapping_rc != 0
        or not npu_ids
        or topology_rc != 0
        or not hccn_tool
        or any(result["rc"] != 0 or not result["ok"] for result in port_results)
    )
    static = {
        "ok": bool(npu_ids) and topology_rc == 0 and topology["ok"] and bool(port_results) and not static_errors,
        "npu_ids": npu_ids,
        "topology": topology,
        "ports": port_results,
        "errors": static_errors,
        "failure_kind": (None if not static_errors else ("capability" if capability_failed else "link")),
    }

    urma_admin = shutil.which("urma_admin")
    endpoints = []
    endpoint_rc = 127
    endpoint_error = "urma_admin is not available"
    if urma_admin:
        endpoint_rc, endpoint_output, endpoint_error = run_cmd([urma_admin, "show"], timeout=15)
        endpoints = parse_urma_endpoints(endpoint_output)
    active_endpoints = [endpoint for endpoint in endpoints if endpoint["link"] == "ACTIVE"]
    urma = {
        "endpoint_rc": endpoint_rc,
        "endpoint_error": redact(endpoint_error),
        "endpoints": endpoints,
        "active_endpoint_count": len(active_endpoints),
        "ok": endpoint_rc == 0 and bool(active_endpoints),
    }

    peer_ping = {
        "requested": bool(context.peer_eid),
        "peer_eid": context.peer_eid,
        "local_eid": active_endpoints[0]["eid"] if active_endpoints else None,
        "ok": False,
    }
    if context.peer_eid and peer_ping["local_eid"] and shutil.which("urma_ping"):
        ping_rc, ping_output, ping_error = run_cmd(
            [
                "urma_ping",
                "-c",
                "1",
                "-W",
                "1",
                "-I",
                peer_ping["local_eid"],
                context.peer_eid,
            ],
            timeout=15,
        )
        peer_ping.update(
            {
                "rc": ping_rc,
                "error": redact(ping_error),
                **parse_urma_ping(
                    ping_output,
                    ping_rc,
                    float(ping_parameters.get("required_packet_loss_percent", 0)),
                ),
            }
        )
    return {"static": static, "urma": urma, "peer_ping": peer_ping}


def probe_hardware(identification: dict[str, Any]) -> dict[str, Any]:
    """Identify the hardware model from npu-smi device names and die topology."""
    sources: list[dict[str, Any]] = []
    rc, out, err = run_cmd(["npu-smi", "info"], timeout=10)
    if out or err:
        sources.append({"source": "npu-smi info", "rc": rc, "output": redact(out or err)})
    topologies = [parse_npu_topology(item["output"]) for item in sources if item["source"] == "npu-smi info"]
    # Confirm hardware only when all visible devices satisfy both name and die-count requirements.
    matched = set()
    match_details: dict[str, Any] = {}
    for topology in topologies:
        for hardware, spec in identification.items():
            names = topology.get("device_names", [])
            name_patterns = spec.get("name_patterns", [])
            names_match = bool(names and name_patterns) and all(
                any(re.fullmatch(str(pattern), name, re.IGNORECASE) for pattern in name_patterns) for name in names
            )
            dies_per_npu = spec.get("dies_per_npu")
            die_counts = topology.get("chips_by_npu", {})
            dies_match = (
                dies_per_npu is None
                or bool(die_counts)
                and all(count == int(dies_per_npu) for count in die_counts.values())
            )
            if names_match and dies_match:
                matched_checks = ["all_device_names"]
                if dies_per_npu is not None:
                    matched_checks.append("dies_per_npu")
                matched.add(hardware.upper())
                match_details[hardware.upper()] = {
                    "source": "npu-smi info",
                    "matched": matched_checks,
                    "device_names": names,
                    "dies_by_npu": die_counts,
                }
    model = next(iter(matched)) if len(matched) == 1 else None
    observed = {
        "device_names": sorted({name for topology in topologies for name in topology.get("device_names", [])}),
        "dies_by_npu": {
            str(npu_id): count for topology in topologies for npu_id, count in topology.get("chips_by_npu", {}).items()
        },
    }
    command_diagnostic = (
        "; ".join(f"{item['source']} rc={item['rc']}" for item in sources) or "No npu-smi/sysfs output was obtained"
    )
    return {
        "model": model,
        "candidates": sorted(matched),
        "sources": compact_sources(sources),
        "match_details": match_details,
        "topologies": topologies,
        "observed": observed,
        "diagnostic": (
            f"{command_diagnostic}; observed identifiers="
            f"{json.dumps(observed, ensure_ascii=False, separators=(',', ':'))}"
        ),
    }


def evaluator_parameters(rules: list[dict[str, Any]], evaluator: str) -> dict[str, Any]:
    return next((rule["parameters"] for rule in rules if rule["evaluator"] == evaluator), {})


PROBES: dict[str, Callable[[CheckContext, list[dict[str, Any]]], dict[str, Any]]] = {
    "hdk": lambda context, rules: {**probe_hdk(), "hardware": context.hardware, "backend": context.backend},
    "kernel": lambda context, rules: {**probe_kernel(), "deployment_mode": context.deployment_mode},
    "memory_scan": lambda context, rules: probe_memory_capacity(context.hardware, context.managed_memory_mb),
    "a5_memory": lambda context, rules: probe_a5_memory_capacity(context.managed_memory_mb),
    "a5_hugepages": lambda context, rules: probe_a5_hugepages(),
    "a5_docker_prereqs": lambda context, rules: probe_docker_prereqs(),
    "a5_secverify": lambda context, rules: probe_a5_secverify(),
    "a5_ub": probe_a5_ub,
    "manual_confirmation": lambda context, rules: {},
    "ssd_storage": lambda context, rules: probe_ssd_storage(context.ssd_path or "", context.backend),
    "roce_diagnostics": lambda context, rules: probe_roce_diagnostics(
        context.hardware,
        context.backend,
        evaluator_parameters(rules, "roce_link"),
        evaluator_parameters(rules, "roce_local"),
    ),
    "hccs_diagnostics": lambda context, rules: probe_hccs_diagnostics(
        context.hardware,
        context.backend,
        evaluator_parameters(rules, "hccs_static"),
        evaluator_parameters(rules, "hccs_dynamic"),
    ),
    "docker_prereqs": lambda context, rules: probe_docker_prereqs(),
    "package_versions": lambda context, rules: probe_package_versions(
        [rule["parameters"]["package"] for rule in rules if rule["parameters"].get("package")]
    ),
    "glibc": lambda context, rules: probe_glibc(),
}


def validate_catalog(catalog: dict[str, Any]) -> None:
    """Validate catalog structure, applicability, and evaluator parameters."""
    # Validate required top-level fields and globally supported values.
    if catalog.get("schema_version") != "7.1":
        raise CatalogError("Only rules.json schema_version=7.1 is supported")
    for field in (
        "supported_hardware",
        "supported_backends",
        "supported_link_types",
        "supported_deployment_modes",
        "rules",
    ):
        if not isinstance(catalog.get(field), list) or not catalog[field]:
            raise CatalogError(f"{field} must be a non-empty array")
    identification = catalog.get("hardware_identification", {})
    if set(catalog["supported_link_types"]) != {"roce", "hccs", "ub"}:
        raise CatalogError("supported_link_types must be roce/hccs/ub")
    if set(catalog["supported_deployment_modes"]) != {"single-node", "multi-node"}:
        raise CatalogError("supported_deployment_modes must be single-node/multi-node")
    if not isinstance(identification, dict) or not set(identification) <= set(catalog["supported_hardware"]):
        raise CatalogError("hardware_identification must map to declared hardware")
    if set(identification) != set(catalog["supported_hardware"]):
        raise CatalogError("hardware_identification must cover every declared hardware type")

    # Validate hardware identification configuration.
    for hardware, spec in identification.items():
        patterns = spec.get("name_patterns") if isinstance(spec, dict) else None
        if (
            not isinstance(patterns, list)
            or not patterns
            or not all(isinstance(pattern, str) and pattern for pattern in patterns)
        ):
            raise CatalogError(f"hardware_identification.{hardware}.name_patterns must be a non-empty string array")
        try:
            for pattern in patterns:
                re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise CatalogError(
                f"hardware_identification.{hardware}.name_patterns contains an invalid regex: {exc}"
            ) from exc
        dies_per_npu = spec.get("dies_per_npu")
        if dies_per_npu is not None and (
            not isinstance(dies_per_npu, int) or isinstance(dies_per_npu, bool) or dies_per_npu <= 0
        ):
            raise CatalogError(f"hardware_identification.{hardware}.dies_per_npu must be a positive integer")
    ids = set()

    # Validate common rule fields, applicability, and remediation definitions.
    for rule in catalog["rules"]:
        required = (
            "id",
            "problem",
            "impact",
            "hardware",
            "backends",
            "scope",
            "blocking",
            "probe",
            "evaluator",
            "parameters",
            "evidence_method",
            "remediation",
        )
        missing = [field for field in required if field not in rule]
        if missing:
            raise CatalogError(f"Rule {rule.get('id', '<unknown>')} is missing fields: {','.join(missing)}")
        if rule["id"] in ids:
            raise CatalogError(f"Duplicate rule ID: {rule['id']}")
        ids.add(rule["id"])
        if (
            not isinstance(rule["hardware"], list)
            or not isinstance(rule["backends"], list)
            or not isinstance(rule["parameters"], dict)
        ):
            raise CatalogError(f"Rule {rule['id']} has invalid hardware/backends/parameters types")
        if rule["scope"] not in ("host", "docker"):
            raise CatalogError(f"Rule {rule['id']} has an invalid scope: {rule['scope']}")
        if rule["probe"] not in PROBES:
            raise CatalogError(f"Rule {rule['id']} uses an unknown probe: {rule['probe']}")
        if rule["evaluator"] not in EVALUATORS:
            raise CatalogError(f"Rule {rule['id']} uses an unknown evaluator: {rule['evaluator']}")
        if not set(rule["hardware"]) - {"*"} <= set(catalog["supported_hardware"]):
            raise CatalogError(f"Rule {rule['id']} uses undeclared hardware")
        if not set(rule["backends"]) - {"*"} <= set(catalog["supported_backends"]):
            raise CatalogError(f"Rule {rule['id']} uses an undeclared backend")
        link_types = rule.get("link_types")
        if link_types is not None and (
            not isinstance(link_types, list)
            or not link_types
            or any(link_type not in catalog["supported_link_types"] for link_type in link_types)
        ):
            raise CatalogError(f"Rule {rule['id']} link_types must use values declared in supported_link_types")
        deployment_modes = rule.get("deployment_modes")
        if deployment_modes is not None and (
            not isinstance(deployment_modes, list)
            or not deployment_modes
            or any(mode not in ("single-node", "multi-node") for mode in deployment_modes)
        ):
            raise CatalogError(f"Rule {rule['id']} deployment_modes must be a non-empty single-node/multi-node array")
        if (
            not isinstance(rule["remediation"], dict)
            or "type" not in rule["remediation"]
            or "description" not in rule["remediation"]
        ):
            raise CatalogError(f"Rule {rule['id']} has invalid remediation")
        positioning = rule.get("positioning")
        if positioning is not None:
            required_positioning = ("layer", "target", "coverage", "excludes")
            if not isinstance(positioning, dict) or any(not positioning.get(key) for key in required_positioning[:-1]):
                raise CatalogError(f"Rule {rule['id']} has invalid positioning")
            if (
                not isinstance(positioning.get("excludes"), list)
                or not positioning["excludes"]
                or not all(isinstance(value, str) and value for value in positioning["excludes"])
            ):
                raise CatalogError(f"Rule {rule['id']} positioning.excludes must be a non-empty string array")
        if rule.get("feature") not in (None, "ssd_offload"):
            raise CatalogError(f"Rule {rule['id']} uses an unknown feature")
        if rule.get("exclude_feature") not in (None, "ssd_offload"):
            raise CatalogError(f"Rule {rule['id']} uses an unknown exclude_feature")
        if rule["evaluator"] in ("hccs_static", "hccs_dynamic") and rule.get("link_types") != ["hccs"]:
            raise CatalogError(f"Rule {rule['id']} must restrict link_types=[hccs]")
        if rule["evaluator"] in ("roce_link", "roce_local") and rule.get("link_types") != ["roce"]:
            raise CatalogError(f"Rule {rule['id']} must restrict link_types=[roce]")
        if rule["evaluator"] in ("a5_ub_static", "a5_urma", "a5_peer_ping") and rule.get("link_types") != ["ub"]:
            raise CatalogError(f"Rule {rule['id']} must restrict link_types=[ub]")
        if rule.get("feature") and rule.get("feature") == rule.get("exclude_feature"):
            raise CatalogError(f"Rule {rule['id']} cannot both require and exclude the same feature")

        # Validate evaluator-specific runtime parameters.
        required_parameters = {
            "hccs_static": (
                "same_card_relation",
                "cross_card_relations",
                "required_health",
                "require_complete_lanes",
                "require_positive_speed",
            ),
            "hccs_dynamic": (
                "collective",
                "elements_per_peer",
                "iterations",
                "timeout_seconds",
                "required_retry_error_delta",
            ),
            "roce_link": ("required_link", "required_health", "require_unique_ip", "require_consistent_netmask_mtu"),
            "roce_local": ("packet_size", "required_packet_loss_percent", "timeout_seconds"),
            "a5_ub_static": ("required_topology_relation", "allowed_port_types", "required_link"),
            "a5_urma": ("require_active_endpoint",),
        }.get(rule["evaluator"], ())
        missing_parameters = [key for key in required_parameters if key not in rule["parameters"]]
        if missing_parameters:
            raise CatalogError(f"Rule {rule['id']} is missing runtime parameters: {','.join(missing_parameters)}")
        if rule["evaluator"] == "hdk_requirement":
            requirements = rule["parameters"].get("requirements")
            if not isinstance(requirements, list) or not requirements:
                raise CatalogError(f"Rule {rule['id']} parameters.requirements must be a non-empty array")
            combinations = []
            for requirement in requirements:
                if not isinstance(requirement, dict) or not all(
                    requirement.get(key) for key in ("hardware", "backend", "minimum")
                ):
                    raise CatalogError(f"Rule {rule['id']} HDK requirements must contain hardware/backend/minimum")
                combination = (requirement["hardware"], requirement["backend"])
                if combination in combinations:
                    raise CatalogError(
                        f"Rule {rule['id']} has a duplicate HDK combination: {combination[0]}/{combination[1]}"
                    )
                combinations.append(combination)
                if requirement["hardware"] not in rule["hardware"] or requirement["backend"] not in rule["backends"]:
                    raise CatalogError(
                        f"Rule {rule['id']} HDK combination is outside its applicability: "
                        f"{combination[0]}/{combination[1]}"
                    )
                minimum = version_tuple(str(requirement["minimum"]))
                if minimum is None:
                    raise CatalogError(
                        f"Rule {rule['id']} has an invalid HDK version threshold: {combination[0]}/{combination[1]}"
                    )
            expected = {(hardware, backend) for hardware in rule["hardware"] for backend in rule["backends"]}
            if set(combinations) != expected:
                raise CatalogError(f"Rule {rule['id']} HDK requirements do not cover every applicable combination")
        if rule["evaluator"] == "docker_prereqs":
            parameters = rule["parameters"]
            if set(parameters) - {"required_all", "required_any"} or "required_all" not in parameters:
                raise CatalogError(
                    f"Rule {rule['id']} Docker parameters must contain required_all and may contain required_any"
                )
            required_all = parameters.get("required_all")
            required_any = parameters.get("required_any", [])
            if (
                not isinstance(required_all, list)
                or not required_all
                or not all(isinstance(key, str) and key for key in required_all)
            ):
                raise CatalogError(f"Rule {rule['id']} parameters.required_all must be a non-empty string array")
            if required_any and (
                not isinstance(required_any, list)
                or not all(
                    isinstance(group, list) and len(group) >= 2 and all(isinstance(key, str) and key for key in group)
                    for group in required_any
                )
            ):
                raise CatalogError(
                    f"Rule {rule['id']} parameters.required_any must contain non-empty groups with at least two keys"
                )
        if rule["evaluator"] == "package_version":
            parameters = rule["parameters"]
            if not isinstance(parameters.get("package"), str) or not parameters["package"]:
                raise CatalogError(f"Rule {rule['id']} parameters.package must be a non-empty string")
            threshold = parameters.get("threshold")
            if threshold is not None and (not isinstance(threshold, str) or not threshold):
                raise CatalogError(f"Rule {rule['id']} parameters.threshold must be a non-empty string")
            exceptions = parameters.get("threshold_exceptions", [])
            if not isinstance(exceptions, list):
                raise CatalogError(f"Rule {rule['id']} parameters.threshold_exceptions must be an array")
            combinations = []
            for exception in exceptions:
                if not isinstance(exception, dict) or set(exception) != {"hardware", "backend", "minimum_hdk"}:
                    raise CatalogError(
                        f"Rule {rule['id']} package threshold exceptions must contain hardware/backend/minimum_hdk"
                    )
                combination = (exception["hardware"], exception["backend"])
                if combination in combinations:
                    raise CatalogError(f"Rule {rule['id']} has a duplicate package threshold exception: {combination}")
                combinations.append(combination)
                if exception["hardware"] not in rule["hardware"] or exception["backend"] not in rule["backends"]:
                    raise CatalogError(
                        f"Rule {rule['id']} package threshold exception is outside its applicability: {combination}"
                    )
                if version_tuple(str(exception["minimum_hdk"])) is None:
                    raise CatalogError(
                        f"Rule {rule['id']} has an invalid package threshold exception HDK: {combination}"
                    )
        if rule["evaluator"] == "hccs_dynamic":
            for key in ("elements_per_peer", "iterations", "timeout_seconds"):
                value = rule["parameters"].get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise CatalogError(f"Rule {rule['id']} parameters.{key} must be a positive integer")
            maximum_delta = rule["parameters"].get("required_retry_error_delta")
            if not isinstance(maximum_delta, int) or isinstance(maximum_delta, bool) or maximum_delta < 0:
                raise CatalogError(
                    f"Rule {rule['id']} parameters.required_retry_error_delta must be a non-negative integer"
                )
            if rule["parameters"].get("collective") != "all_to_all_single":
                raise CatalogError(f"Rule {rule['id']} supports only collective=all_to_all_single")
        if rule["evaluator"] == "roce_local":
            timeout_seconds = rule["parameters"].get("timeout_seconds")
            packet_loss = rule["parameters"].get("required_packet_loss_percent")
            if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
                raise CatalogError(f"Rule {rule['id']} parameters.timeout_seconds must be a positive integer")
            if not isinstance(packet_loss, (int, float)) or isinstance(packet_loss, bool) or packet_loss < 0:
                raise CatalogError(f"Rule {rule['id']} parameters.required_packet_loss_percent must be non-negative")
    # Validate probe/evaluator pairings.
    for probe_name, required_evaluators in {
        "roce_diagnostics": {"roce_link", "roce_local"},
        "hccs_diagnostics": {"hccs_static", "hccs_dynamic"},
        "a5_ub": {"a5_ub_static", "a5_urma", "a5_peer_ping"},
        "a5_docker_prereqs": {"docker_prereqs"},
    }.items():
        configured = {rule["evaluator"] for rule in catalog["rules"] if rule["probe"] == probe_name}
        if configured and configured != required_evaluators:
            raise CatalogError(
                f"probe {probe_name} must configure both evaluators: {','.join(sorted(required_evaluators))}"
            )


def applies(rule: dict[str, Any], hardware: str, backend: str, link_type: str, deployment_mode: str) -> bool:
    link_types = rule.get("link_types")
    deployment_modes = rule.get("deployment_modes")
    return (
        ("*" in rule["hardware"] or hardware in rule["hardware"])
        and ("*" in rule["backends"] or backend in rule["backends"])
        and (not link_types or link_type in link_types)
        and (not deployment_modes or deployment_mode in deployment_modes)
    )


def feature_applies(rule: dict[str, Any], ssd_offload: bool) -> bool:
    enabled_features = {"ssd_offload"} if ssd_offload else set()
    required = rule.get("feature")
    excluded = rule.get("exclude_feature")
    return (not required or required in enabled_features) and (not excluded or excluded not in enabled_features)


def validate_inputs(
    catalog: dict[str, Any], hardware_arg: str, backend_arg: str, link_type_arg: str, deployment_mode_arg: str
) -> tuple[str, str, str, str, dict[str, Any]]:
    """Validate CLI inputs and confirm that --hardware matches the detected hardware."""
    backend = backend_arg.strip().lower()
    if backend not in catalog["supported_backends"]:
        raise InputError(
            f"--backend={backend_arg!r} is outside the supported set {','.join(catalog['supported_backends'])}"
        )
    hardware = hardware_arg.strip().upper()
    if hardware not in catalog["supported_hardware"]:
        raise InputError(
            f"--hardware={hardware_arg!r} is outside the supported set {','.join(catalog['supported_hardware'])}"
        )
    link_type = link_type_arg.strip().lower()
    if link_type not in catalog["supported_link_types"]:
        raise InputError(
            f"--link-type={link_type_arg!r} is outside the supported set {','.join(catalog['supported_link_types'])}"
        )
    deployment_mode = deployment_mode_arg.strip().lower()
    if deployment_mode not in catalog["supported_deployment_modes"]:
        raise InputError(
            f"--deployment-mode={deployment_mode_arg!r} is outside the supported set "
            f"{','.join(catalog['supported_deployment_modes'])}"
        )
    if hardware == "A2" and link_type != "roce":
        raise InputError("A2 supports only --link-type roce")
    if hardware == "A3" and link_type not in ("roce", "hccs"):
        raise InputError("A3 supports --link-type roce or hccs")
    if hardware == "A5" and link_type != "ub":
        raise InputError("950PR&950DT Products support only --link-type ub")
    detected = probe_hardware(catalog.get("hardware_identification", {}))
    if not detected["model"]:
        detail = ", ".join(detected["candidates"]) if detected["candidates"] else "No supported model was detected"
        raise InputError(
            f"Cannot confirm the current hardware model ({detail}); evidence: {detected['diagnostic']}. "
            "Refusing to continue"
        )
    if detected["model"] != hardware:
        raise InputError(f"--hardware={hardware} does not match the detected hardware {detected['model']}")
    return (
        hardware,
        backend,
        link_type,
        deployment_mode,
        {
            "requested": hardware,
            "detected": detected["model"],
            "identification": {
                "match_details": detected.get("match_details", {}),
                "topologies": detected.get("topologies", []),
            },
            "sources": detected["sources"],
            "supported": catalog["supported_hardware"],
        },
    )


def make_record(rule: dict[str, Any], status: str, evidence: Any, detail: str) -> dict[str, Any]:
    remediation = dict(rule["remediation"])
    record = {
        "id": rule["id"],
        "status": status,
        "problem": rule["problem"],
        "impact": rule["impact"],
        "detail": detail,
        "evidence_method": rule["evidence_method"],
        "evidence": evidence,
        "applicability": {
            "hardware": rule["hardware"],
            "backends": rule["backends"],
            "link_types": rule.get("link_types"),
            "deployment_modes": rule.get("deployment_modes"),
            "scope": rule["scope"],
        },
        "blocking": rule["blocking"],
        "remediation": remediation,
    }
    if "positioning" in rule:
        record["positioning"] = copy.deepcopy(rule["positioning"])
    return record


def is_blocking_result(result: dict[str, Any]) -> bool:
    return bool(result["blocking"] and result["status"] in ("BLOCK", "MANUAL_REQUIRED", "NOT_EXECUTED"))


def compare_version(value: tuple[int, ...], threshold: tuple[int, ...]) -> int:
    width = max(len(value), len(threshold))
    left = value + (0,) * (width - len(value))
    right = threshold + (0,) * (width - len(threshold))
    return (left > right) - (left < right)


def build_version_evidence(rule: dict[str, Any], probe: dict[str, Any], raw_version: str | None) -> dict[str, Any]:
    value_key = rule["parameters"].get("value_key", "version")
    if value_key == "cann_version":
        source = probe.get("cann_source")
    elif rule["probe"] == "hdk":
        source = probe.get("version_source")
    elif rule["probe"] == "kernel":
        source = {"source": "uname -r", "rc": probe.get("rc"), "stderr": probe.get("stderr")}
    elif rule["probe"] == "glibc":
        source = {"source": "ldd --version", "rc": probe.get("rc"), "summary": probe.get("summary")}
    else:
        source = None
    return {
        "raw_version": raw_version,
        "normalized_version": version_text(raw_version or "") if raw_version else None,
        "source": source,
    }


def evaluate_version(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any] | None:
    """Evaluate a generic version dependency using the rule's comparison condition."""
    params = rule["parameters"]
    probe = probes[rule["probe"]]
    value_text = str(probe.get(params.get("value_key", "version")) or "")
    value = version_tuple(value_text)
    evidence = build_version_evidence(rule, probe, value_text or None)
    if value is None:
        if params.get("suppress_if_unknown"):
            return None
        detail = probe.get("query_summary", "No parseable version field was obtained")
        return make_record(
            rule,
            params.get("unknown_status", "BLOCK"),
            evidence,
            f"Cannot confirm the version; {detail}. Unknown evidence cannot be treated as a pass",
        )
    if "threshold" in params:
        comparison = compare_version(value, version_tuple(params["threshold"]) or (0,))
        problem_hit = comparison < 0 if params["operator"] == "lt" else comparison >= 0
    else:
        minimum = compare_version(value, version_tuple(params["min"]) or (0,))
        maximum = compare_version(value, version_tuple(params["max"]) or (0,))
        problem_hit = (minimum >= 0 if params.get("min_inclusive", True) else minimum > 0) and (
            maximum < 0 if params.get("max_exclusive", True) else maximum <= 0
        )
    status = ("BLOCK" if rule["blocking"] else "WARN") if problem_hit else "PASS"
    normalized = version_text(value_text)
    display = f"Detected version={value_text}" + (f" (normalized={normalized})" if value_text != normalized else "")
    return make_record(rule, status, evidence, display)


def evaluate_kernel_requirement(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the Linux kernel dependency for the deployment mode and threshold."""
    probe = probes[rule["probe"]]
    deployment_mode = probe.get("deployment_mode")
    required_deployment_modes = rule.get("deployment_modes", [])
    evidence = {**probe, "deployment_mode": deployment_mode, "required_deployment_modes": required_deployment_modes}
    if deployment_mode not in required_deployment_modes:
        result = make_record(
            rule,
            "NOT_APPLICABLE",
            evidence,
            "The current deployment is single-node; KERNEL-001 does not affect single-node KV Pool",
        )
        result["blocking"] = False
        result["remediation"] = dict(rule["remediation"])
        result["remediation"]["description"] = "No action is required for single-node deployments."
        return result
    value_text = str(probe.get("value") or "")
    value = version_tuple(value_text)
    evidence.update(build_version_evidence(rule, probe, value_text or None))
    threshold_text = str(rule["parameters"]["threshold"])
    if value is None:
        return make_record(
            rule,
            rule["parameters"].get("unknown_status", "BLOCK"),
            evidence,
            (
                f"Cannot confirm the Linux kernel version for a multi-node deployment; "
                f"required>={threshold_text}. Unknown evidence cannot be treated as a pass"
            ),
        )
    status = "PASS" if compare_version(value, version_tuple(threshold_text) or (0,)) >= 0 else "BLOCK"
    detail = f"Deployment mode={deployment_mode}; detected Linux kernel={value_text}, required>={threshold_text}"
    result = make_record(rule, status, evidence, detail)
    if status == "BLOCK":
        result["remediation"]["description"] = (
            f"For multi-node deployments, upgrade the host Linux kernel to >={threshold_text}, "
            "restart or rebuild the container, and rerun the check."
        )
    return result


def evaluate_hdk_requirement(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the minimum HDK version for the hardware/backend matrix."""
    probe = probes[rule["probe"]]
    hardware = probe.get("hardware")
    backend = probe.get("backend")
    requirement = next(
        (
            item
            for item in rule["parameters"]["requirements"]
            if item["hardware"] == hardware and item["backend"] == backend
        ),
        None,
    )
    if requirement is None:
        evidence = {"hardware": hardware, "backend": backend, "requirement": None}
        return make_record(
            rule,
            "BLOCK",
            evidence,
            "No HDK version requirement exists for the current hardware/backend, so a pass cannot be determined",
        )

    raw_version = str(probe.get("version") or "")
    current = version_tuple(raw_version)
    minimum_text = str(requirement["minimum"])
    minimum = version_tuple(minimum_text) or (0,)
    evidence = {
        **build_version_evidence(rule, probe, raw_version or None),
        "hardware": hardware,
        "backend": backend,
        "minimum_version": minimum_text,
    }
    if current is None:
        return make_record(
            rule,
            rule["parameters"].get("unknown_status", "BLOCK"),
            evidence,
            (
                f"Cannot confirm the HDK version; the current combination requires >={minimum_text}. "
                "Unknown evidence cannot be treated as a pass"
            ),
        )

    status = "PASS" if compare_version(current, minimum) >= 0 else "BLOCK"
    normalized = version_text(raw_version)
    detected = raw_version + (f" (normalized={normalized})" if raw_version != normalized else "")
    detail = f"Detected HDK={detected}; {hardware}/{backend} requires >={minimum_text}"
    result = make_record(rule, status, evidence, detail)
    if status == "BLOCK":
        result["remediation"]["description"] = (
            f"Upgrade the host HDK to >={minimum_text}, refresh Docker mounts, "
            "restart the container, and rerun the check."
        )
    return result


def evaluate_memory_scan(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate managed memory using A2 static huge-page or A3 mem_scan evidence."""
    probe = probes["memory_scan"]
    status = "PASS" if probe.get("ok") else "BLOCK"
    target_mb = probe.get("managed_memory_mb")
    available_mb = probe.get("available_memory_mb")
    if probe.get("method") == "a2_static_hugepages":
        detail = (
            f"Method=2MB static huge pages; global target={target_mb}MB; "
            f"page size={probe.get('page_size_kb') or '-'}kB; "
            f"total pages={probe.get('total_pages') if probe.get('total_pages') is not None else '-'}; "
            f"free pages={probe.get('free_pages') if probe.get('free_pages') is not None else '-'}; "
            f"reserved pages={probe.get('reserved_pages') if probe.get('reserved_pages') is not None else '-'}; "
            f"allocatable pages={probe.get('available_pages') if probe.get('available_pages') is not None else '-'}; "
            f"required pages={probe.get('required_pages')}; "
            f"available capacity={available_mb if available_mb is not None else '-'}MB"
        )
        if probe.get("error"):
            detail += f"; error={probe['error']}"
        if status == "BLOCK" and probe.get("failure_kind") == "capacity" and available_mb is not None:
            shortage_mb = target_mb - available_mb
            record_fix = (
                f"A2 is short of 2MB static huge pages (target {target_mb}MB, available {available_mb}MB, "
                f"shortage {shortage_mb:.0f}MB). Configure or expand HugePages on the host, then rerun the check."
            )
        elif status == "BLOCK":
            record_fix = (
                "Check /proc/meminfo visible in the container and confirm Hugepagesize=2048kB "
                "and readable static huge-page counters. Repair the environment and rerun the check."
            )
        else:
            record_fix = rule["remediation"]["description"]
        result = make_record(rule, status, probe, detail)
        result["remediation"]["description"] = record_fix
        return result
    primary_mb = (probe.get("scans", {}).get("1024") or {}).get("total_available_mb")
    fallback_mb = (probe.get("scans", {}).get("2") or {}).get("total_available_mb")
    detail = (
        f"script={probe.get('script')}; global target={target_mb}MB; "
        f"1GB scan available={primary_mb if primary_mb is not None else '-'}MB; "
        f"2MB scan available={fallback_mb if fallback_mb is not None else '-'}MB; "
        f"matched granularity={probe.get('matched_granularity_mb') or '-'}MB"
    )
    if probe.get("error"):
        detail += f"; error={probe['error']}"
    if status == "BLOCK" and probe.get("failure_kind") == "capacity":
        shortage_mb = (
            target_mb - available_mb
            if isinstance(target_mb, (int, float)) and isinstance(available_mb, (int, float))
            else None
        )
        shortage_text = f", shortage {shortage_mb:.0f}MB" if shortage_mb is not None else ""
        record_fix = (
            f"Insufficient global available capacity "
            f"(target {target_mb}MB, available {available_mb}MB{shortage_text}). "
            "Confirm the target, release memory, and rerun the check. "
            "If capacity is still insufficient, drain workloads and restart the host."
        )
    elif status == "BLOCK":
        record_fix = (
            "The memory scan did not succeed. Check Docker permissions, scan-script dependencies, "
            "and kernel interfaces. Repair the environment and rerun the check. "
            "If it cannot be restored, follow the maintenance process to drain workloads and restart the host."
        )
    else:
        record_fix = rule["remediation"]["description"]
    if probe.get("diagnostic"):
        detail += f"; error={probe['diagnostic']}"
    result = make_record(rule, status, probe, detail)
    result["remediation"]["description"] = record_fix
    return result


def evaluate_docker_prereqs(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether required Docker devices, drivers, and HCCN entry points are available."""
    probe = probes[rule["probe"]]
    missing_all = [key for key in rule["parameters"]["required_all"] if not probe.get(key)]
    failed_any = [
        group for group in rule["parameters"].get("required_any", []) if not any(probe.get(key) for key in group)
    ]
    status = "PASS" if not missing_all and not failed_any else "BLOCK"
    hccn_entry = "hccn_conf" if probe.get("hccn_conf") else ("hccn_tool" if probe.get("hccn_tool") else None)
    evidence = {
        **probe,
        "satisfied_hccn_entry": hccn_entry,
        "missing_required_all": missing_all,
        "failed_required_any": failed_any,
    }
    failures = list(missing_all) + [" or ".join(group) for group in failed_any]
    detail = (
        "Missing: " + ", ".join(failures)
        if failures
        else f"Docker devices and driver are visible; HCCN entry point={hccn_entry}"
    )
    result = make_record(rule, status, evidence, detail)
    if failures:
        result["remediation"]["description"] = (
            f"Recreate the container and provide the following devices, mounts, or tools: "
            f"{', '.join(failures)}. Then rerun the check."
        )
    return result


def evaluate_manual_confirmation(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Generate a manual-confirmation result when an external device version cannot be read automatically."""
    evidence = {
        "automatic_detection": False,
        "required_confirmation": rule["parameters"].get("required_confirmation", rule["evidence_method"]),
    }
    detail = (
        "The tool does not probe switch versions; operations must confirm the version before deployment can continue"
    )
    return make_record(rule, "MANUAL_REQUIRED", evidence, detail)


def evaluate_package_version(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether a required package is installed and meets its version threshold."""
    name = rule["parameters"]["package"]
    version = probes["package_versions"]["versions"].get(name)
    threshold = rule["parameters"].get("threshold")
    hdk_probe = probes.get("hdk", {})
    detected_hdk = str(hdk_probe.get("version") or "")
    threshold_exception = None
    for exception in rule["parameters"].get("threshold_exceptions", []):
        if hdk_probe.get("hardware") != exception["hardware"] or hdk_probe.get("backend") != exception["backend"]:
            continue
        current_hdk = version_tuple(detected_hdk)
        minimum_hdk = version_tuple(exception["minimum_hdk"])
        if current_hdk is not None and minimum_hdk is not None and compare_version(current_hdk, minimum_hdk) >= 0:
            threshold = None
            threshold_exception = exception
            break
    evidence = {
        "package": name,
        "installed_version": version,
        "required_version": threshold,
        "threshold_exception": threshold_exception,
        "detected_hdk": detected_hdk or None,
    }
    if not version:
        return make_record(rule, "BLOCK", evidence, f"{name} is not installed")
    if not threshold:
        detail = f"{name}={version} is installed"
        if threshold_exception:
            detail += (
                f"; {threshold_exception['hardware']}/{threshold_exception['backend']} with "
                f"HDK>={threshold_exception['minimum_hdk']} has no minimum package version"
            )
        return make_record(rule, "PASS", evidence, detail)
    current_key = package_version_key(version)
    threshold_key = package_version_key(threshold)
    if current_key is None or threshold_key is None:
        return make_record(rule, "BLOCK", evidence, f"Cannot parse the version of {name}: {version}")
    status = "BLOCK" if current_key < threshold_key else "PASS"
    return make_record(rule, status, evidence, f"Detected {name}={version}, required>={threshold}")


def evaluate_ssd_storage(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Offload storage conditions for the current backend using SSD probe evidence."""
    probe = probes["ssd_storage"]
    status = "PASS" if probe.get("ok") else "BLOCK"
    detail = f"SSD path={probe.get('path')}; backend={probe.get('backend')}"
    if probe.get("error"):
        detail += f"; error={probe['error']}"
    return make_record(rule, status, probe, detail)


def evaluate_roce_link(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate device-side link readiness from static RoCE NPU port state."""
    probe = probes["roce_diagnostics"]
    link = probe["link"]
    status = "PASS" if link.get("ok") else "BLOCK"
    evidence = {
        "hardware": probe["hardware"],
        "backend": probe["backend"],
        "available_npu_ids": probe.get("available_npu_ids", []),
        "npu_states": probe.get("npu_states", []),
        "failure_kind": link.get("failure_kind"),
        "error": link.get("error"),
    }
    detail = (
        f"Visible NPUs={len(evidence['available_npu_ids'])}; per-NPU static port readiness="
        f"{'PASS' if status == 'PASS' else 'FAIL'}"
    )
    if link.get("error"):
        detail += f"; error={link['error']}"
    result = make_record(rule, status, evidence, detail)
    if status == "BLOCK" and link.get("failure_kind") == "capability":
        result["remediation"]["description"] = (
            "Check hccn_tool, NPU device permissions, and driver mounts in the container. "
            "Restore per-NPU IP/Link/health/MTU queries and retry."
        )
    return result


def evaluate_roce_local(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate dynamic RoCE paths from local directed-pair ping results and error-counter deltas."""
    probe = probes["roce_diagnostics"]
    local = probe["local"]
    if not local.get("executed", True) and local.get("failure_kind") == "static_check_failed":
        status = "NOT_EXECUTED"
    else:
        status = "PASS" if local.get("ok") else "BLOCK"
    evidence = {
        "hardware": probe["hardware"],
        "backend": probe["backend"],
        "available_npu_ids": probe.get("available_npu_ids", []),
        **local,
    }
    failed_count = sum(1 for item in local.get("directions", []) if not item.get("ok"))
    if status == "NOT_EXECUTED":
        detail = "Static port checks failed; local NPU-to-NPU data-path tests were not run"
    else:
        counter_deltas = local.get("error_counters", {}).get("deltas", {})
        counter_increase_count = sum(
            1 for counters in counter_deltas.values() for value in counters.values() if value > 0
        )
        detail = (
            f"Directed pairs={local.get('direction_count', len(local.get('directions', [])))}; "
            f"near-MTU packet size={local.get('packet_size')}B; failures={failed_count}; "
            f"new error counters={counter_increase_count}"
        )
    if local.get("error"):
        detail += f"; error={local['error']}"
    result = make_record(rule, status, evidence, detail)
    if status == "NOT_EXECUTED":
        result["blocking"] = False
        result["remediation"]["description"] = "Make all NPU ports statically ready, then rerun the check."
    if status == "BLOCK" and local.get("failure_kind") in ("capability", "command"):
        result["remediation"]["description"] = (
            "The local NPU-to-NPU test ran but evidence collection was incomplete. "
            "Check hccn_tool execution capability and error-counter query permissions, then retry. "
            "A command error must not be treated as network failure."
        )
    return result


def evaluate_hccs_static(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate intra-node links from HCCS topology, health, lane, and speed evidence."""
    static = probes["hccs_diagnostics"]["static"]
    status = "PASS" if static.get("ok") else "BLOCK"
    detail = (
        f"Visible NPUs={len(static.get('visible_npu_ids', []))}; "
        f"visible dies={len(static.get('device_pairs', []))}; "
        f"SIO/HCCS static check={'PASS' if status == 'PASS' else 'FAIL'}"
    )
    if static.get("error"):
        detail += f"; error={static['error']}"
    result = make_record(rule, status, static, detail)
    if status == "BLOCK" and static.get("failure_kind") == "capability":
        result["remediation"]["description"] = (
            "Confirm Docker exposes all A3 NPUs, npu-smi supports topo/hccs queries, "
            "and driver mounts are complete. Restore static diagnostics and rerun the check."
        )
    return result


def evaluate_hccs_dynamic(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate dynamic interconnect state from HCCL tests and HCCS error-counter deltas."""
    dynamic = probes["hccs_diagnostics"]["dynamic"]
    if not dynamic.get("executed", True):
        status = "NOT_EXECUTED"
    else:
        status = "PASS" if dynamic.get("ok") else "BLOCK"
    detail = (
        f"HCCL all-to-all={'NOT_EXECUTED' if status == 'NOT_EXECUTED' else ('PASS' if status == 'PASS' else 'FAIL')}"
    )
    if dynamic.get("world_size") is not None:
        detail += f"; ranks={dynamic['world_size']}"
    if dynamic.get("executed"):
        deltas = dynamic.get("error_counters", {}).get("deltas", {})
        increase_count = sum(
            1 for counters in deltas.values() for values in counters.values() for value in values if value > 0
        )
        detail += f"; new retry/error counters={increase_count}"
    if dynamic.get("error"):
        detail += f"; reason={dynamic['error']}"
    result = make_record(rule, status, dynamic, detail)
    if status == "NOT_EXECUTED" and dynamic.get("failure_kind") == "static_check_failed":
        result["blocking"] = False
        result["remediation"]["description"] = "Make all A3 SIO/HCCS static links ready, then rerun the check."
    elif status == "NOT_EXECUTED" and dynamic.get("failure_kind") == "devices_busy":
        result["remediation"]["description"] = (
            "Drain or migrate current NPU workloads, then rerun the check after all visible NPUs are idle. "
            "The tool does not forcibly terminate workloads."
        )
    elif status != "PASS" and dynamic.get("failure_kind") == "capability":
        result["remediation"]["description"] = (
            "Confirm Python, torch_npu, and HCCL are available in the container, all A3 dies are exposed, "
            "and no stale test processes remain. Restore dynamic-check capability and rerun the check."
        )
    return result


def evaluate_a5_memory(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate 950PR&950DT Products host memory capacity."""
    probe = probes["a5_memory"]
    status = "PASS" if probe.get("ok") else "BLOCK"
    detail = (
        f"Method=950PR&950DT Products host memory; global target={probe.get('managed_memory_mb')}MB; "
        f"MemTotal={probe.get('total_memory_mb') if probe.get('total_memory_mb') is not None else '-'}MB; "
        f"MemAvailable={probe.get('available_memory_mb') if probe.get('available_memory_mb') is not None else '-'}MB"
    )
    if probe.get("error"):
        detail += f"; error={probe['error']}"
    result = make_record(rule, status, probe, detail)
    if status == "BLOCK" and probe.get("failure_kind") == "capacity":
        result["remediation"]["description"] = (
            "Reduce the configured backend memory target or release host memory, then rerun the check."
        )
    elif status == "BLOCK":
        result["remediation"]["description"] = (
            "Confirm MemTotal and MemAvailable are readable from /proc/meminfo, repair the container, "
            "and rerun the check."
        )
    return result


def evaluate_a5_hugepages(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate 950PR&950DT Products huge-page cleanup status."""
    probe = probes["a5_hugepages"]
    if not probe.get("evidence_complete"):
        status = "BLOCK"
    elif probe.get("residual_entries"):
        status = "WARN"
    else:
        status = "PASS"
    residual = probe.get("residual_entries", [])
    detail = (
        f"950PR&950DT Products huge-page entries read={len(probe.get('entries', []))}; residual entries={len(residual)}"
    )
    if residual:
        detail += "; residual=" + ", ".join(
            f"node{item['node']}/{item['page_size_kb']}kB={item['configured_pages']}" for item in residual
        )
    if probe.get("errors"):
        detail += "; errors=" + " | ".join(probe["errors"])
    return make_record(rule, status, probe, detail)


def evaluate_a5_secverify(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the documented 950PR&950DT Products custom-op signature state."""
    probe = probes["a5_secverify"]
    required_enable = bool(rule["parameters"]["required_enable"])
    required_mode = int(rule["parameters"]["required_mode"])
    invalid = [
        state
        for state in probe.get("states", [])
        if state.get("enable") != required_enable or state.get("mode") != required_mode
    ]
    if probe.get("errors"):
        status = "MANUAL_REQUIRED"
    elif invalid:
        status = "BLOCK"
    else:
        status = "PASS"
    detail = f"Visible NPUs={len(probe.get('npu_ids', []))}; mismatches={len(invalid)}"
    if invalid:
        detail += "; states=" + json.dumps(
            [
                {"npu_id": state.get("npu_id"), "enable": state.get("enable"), "mode": state.get("mode")}
                for state in invalid
            ],
            separators=(",", ":"),
        )
    if probe.get("errors"):
        detail += "; errors=" + " | ".join(probe["errors"])
    result = make_record(rule, status, probe, detail)
    if status == "MANUAL_REQUIRED":
        result["remediation"]["description"] = (
            "Run the checker in a privileged or NPU-exclusive container, or manually confirm "
            "custom-op-secverify-enable and custom-op-secverify-mode on every visible 950PR&950DT Products NPU."
        )
    elif status == "BLOCK":
        result["remediation"]["description"] = (
            "Correct custom-op-secverify-enable or custom-op-secverify-mode to the documented value, "
            "or confirm the 950PR&950DT Products signature state with the hardware administrator."
        )
    return result


def evaluate_a5_ub_static(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate 950PR&950DT Products NPU topology and physical UB port state."""
    static = probes["a5_ub"]["static"]
    status = "PASS" if static.get("ok") else "BLOCK"
    detail = (
        f"Visible NPUs={len(static.get('npu_ids', []))}; "
        f"topology={'PASS' if static.get('topology', {}).get('ok') else 'FAIL'}; "
        f"port queries={sum(1 for item in static.get('ports', []) if item.get('ok'))}"
    )
    if static.get("errors"):
        detail += "; errors=" + " | ".join(static["errors"])
    result = make_record(rule, status, static, detail)
    if status == "BLOCK" and static.get("failure_kind") == "capability":
        result["remediation"]["description"] = (
            "Run the 950PR&950DT Products check in an NPU-exclusive container with npu-smi and hccn_tool mounted, "
            "then rerun the check."
        )
    elif status == "BLOCK":
        result["remediation"]["description"] = (
            "Inspect the failed 950PR&950DT Products UDie and port, restore the UB/UBOE physical link, "
            "and rerun the check."
        )
    return result


def evaluate_a5_urma(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate 950PR&950DT Products URMA endpoints."""
    urma = probes["a5_ub"]["urma"]
    status = "PASS" if urma.get("ok") else "BLOCK"
    detail = f"Active endpoints={urma.get('active_endpoint_count', 0)}; urma_admin show rc={urma.get('endpoint_rc')}"
    if urma.get("endpoint_error"):
        detail += f"; endpoint error={urma['endpoint_error']}"
    result = make_record(rule, status, urma, detail)
    if status == "BLOCK":
        result["remediation"]["description"] = (
            "Restore at least one ACTIVE URMA endpoint and verify /etc/hixlep and /etc/hccl_rootinfo.json, "
            "then rerun the check."
        )
    return result


def evaluate_a5_peer_ping(rule: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an optional cross-node 950PR&950DT Products URMA ping."""
    peer = probes["a5_ub"]["peer_ping"]
    if not peer.get("requested"):
        result = make_record(
            rule,
            "NOT_EXECUTED",
            peer,
            "No --peer-eid was provided; cross-node URMA connectivity was not tested",
        )
        result["blocking"] = False
        return result
    status = "PASS" if peer.get("ok") else "BLOCK"
    detail = (
        f"peer_eid={peer.get('peer_eid')}; local_eid={peer.get('local_eid')}; "
        f"transmitted={peer.get('transmitted')}; received={peer.get('received')}; "
        f"packet_loss={peer.get('packet_loss_percent')}%"
    )
    if peer.get("error"):
        detail += f"; error={peer['error']}"
    result = make_record(rule, status, peer, detail)
    result["blocking"] = status == "BLOCK"
    return result


def package_version_key(value: str) -> tuple[int, ...] | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?", value, re.IGNORECASE)
    if not match:
        return None
    release = tuple(int(item) for item in match.group(1).split("."))
    release = (release + (0, 0, 0, 0))[:4]
    prerelease = match.group(2)
    if prerelease:
        stage = {"a": -3, "b": -2, "rc": -1}[prerelease.lower()]
        serial = int(match.group(3) or 0)
    elif match.group(4) is not None:
        stage = 1
        serial = int(match.group(4))
    else:
        stage = 0
        serial = 0
    return release + (stage, serial)


EVALUATORS: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None]] = {
    "version_range": evaluate_version,
    "kernel_requirement": evaluate_kernel_requirement,
    "hdk_requirement": evaluate_hdk_requirement,
    "memory_scan": evaluate_memory_scan,
    "docker_prereqs": evaluate_docker_prereqs,
    "manual_confirmation": evaluate_manual_confirmation,
    "package_version": evaluate_package_version,
    "ssd_storage": evaluate_ssd_storage,
    "roce_link": evaluate_roce_link,
    "roce_local": evaluate_roce_local,
    "hccs_static": evaluate_hccs_static,
    "hccs_dynamic": evaluate_hccs_dynamic,
    "a5_memory": evaluate_a5_memory,
    "a5_hugepages": evaluate_a5_hugepages,
    "a5_secverify": evaluate_a5_secverify,
    "a5_ub_static": evaluate_a5_ub_static,
    "a5_urma": evaluate_a5_urma,
    "a5_peer_ping": evaluate_a5_peer_ping,
}


def build_probe_plan(rules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group applicable Docker rules by probe for shared execution."""
    plan: dict[str, list[dict[str, Any]]] = {}
    for rule in rules:
        if rule["scope"] == "docker":
            plan.setdefault(rule["probe"], []).append(rule)
    return plan


PROBE_LABELS = {
    "memory_scan": "managed memory check",
    "a5_memory": "950PR&950DT Products host memory check",
    "a5_hugepages": "950PR&950DT Products huge-page check",
    "a5_docker_prereqs": "950PR&950DT Products device and configuration mount check",
    "a5_secverify": "950PR&950DT Products custom-op signature state check",
    "a5_ub": "950PR&950DT Products UB and URMA check",
    "hdk": "version check",
    "kernel": "Linux kernel version check",
    "manual_confirmation": "manual confirmation preparation",
    "hccs_diagnostics": "A3 intra-node interconnect check",
    "roce_diagnostics": "RoCE check",
    "docker_prereqs": "Docker device and mount check",
    "package_versions": "backend package check",
    "glibc": "glibc version check",
    "ssd_storage": "SSD Offload storage check",
}


def collect_check_results(catalog: dict[str, Any], context: CheckContext) -> list[dict[str, Any]]:
    """Filter applicable rules, run shared probes, and build scenario results."""
    # Select rules applicable to the hardware, backend, link, deployment mode, and feature state.
    applicable_rules = [
        rule
        for rule in catalog["rules"]
        if applies(rule, context.hardware, context.backend, context.link_type, context.deployment_mode)
        and feature_applies(rule, context.ssd_offload)
    ]

    # Group rules by probe so multiple rules share each low-level probe.
    probe_plan = build_probe_plan(applicable_rules)
    progress = context.progress or ProgressReporter(False)
    probe_results = {}
    total = len(probe_plan)
    for index, (name, probe_rules) in enumerate(probe_plan.items(), start=1):
        rule_ids = ",".join(rule["id"] for rule in probe_rules)
        label = f"{rule_ids} {PROBE_LABELS.get(name, name)}"
        with progress.stage(label, index, total):
            probe_results[name] = PROBES[name](context, probe_rules)

    # Convert probe results into scenario results in catalog order.
    check_results = []
    for rule in applicable_rules:
        result: dict[str, Any] | None
        if rule["scope"] == "host":
            result = make_record(
                rule,
                "NOT_EXECUTED",
                {},
                "The current tool supports Docker only; this host-scope scenario is not executed in the container",
            )
        else:
            result = EVALUATORS[rule["evaluator"]](rule, probe_results)
        if result is not None:
            check_results.append(result)
    return check_results


def is_memory_capacity_shortage(record: dict[str, Any]) -> bool:
    evidence = record.get("evidence", {})
    return (
        record.get("id") == "MEM-001"
        and record.get("status") == "BLOCK"
        and evidence.get("failure_kind") == "capacity"
        and evidence.get("method") == "a3_mem_scan"
    )


def fix_memory_with_drop_caches(
    rule: dict[str, Any], context: CheckContext, record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop caches and recheck memory capacity when A3 capacity repair conditions are met."""
    action: dict[str, Any] = {
        "id": record["id"],
        "type": "drop_caches_then_rescan",
        "status": "SKIPPED",
        "message": "",
        "before_evidence": record.get("evidence", {}),
        "operation": {"sync": None, "drop_caches": {"path": "/proc/sys/vm/drop_caches", "value": 3}},
        "after_evidence": None,
    }
    # Allow Drop Cache only after an A3 scan confirms insufficient capacity; otherwise require manual handling.
    if not is_memory_capacity_shortage(record):
        if record.get("evidence", {}).get("method") == "a2_static_hugepages":
            action["type"] = "manual_static_hugepages"
            action["message"] = (
                "A2 static 2MB huge pages are insufficient or detection capability is unavailable. "
                "Drop Cache does not increase static huge pages and was not executed. "
                "Configure or expand HugePages on the host and rerun the check."
            )
        else:
            action["message"] = (
                "The memory check did not produce A3 capacity evidence that permits Drop Cache, "
                "so it was not executed. Restore detection capability and rerun the check."
            )
        return record, action
    if os.geteuid() != 0:
        action.update(
            status="FAILED",
            message=(
                "Drop Cache requires root privileges in the container and was not executed. "
                "Adjust permissions and rerun the check. "
                "If this cannot be resolved, drain workloads and restart the host."
            ),
        )
        return record, action

    sync_rc, _sync_out, sync_err = run_cmd(["sync"], timeout=60)
    action["operation"]["sync"] = {"rc": sync_rc, "stderr": redact(sync_err)}
    if sync_rc != 0:
        action.update(
            status="FAILED",
            message=(
                f"sync failed, so Drop Cache was not executed: {redact(sync_err)}. "
                "Repair the issue and rerun the check. "
                "If this cannot be resolved, drain workloads and restart the host."
            ),
        )
        return record, action
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="ascii")
    except OSError as exc:
        action.update(
            status="FAILED",
            message=(
                f"Cannot write to /proc/sys/vm/drop_caches, so the repair did not complete: {exc}. "
                "Adjust container permissions and rerun the check. "
                "If this cannot be resolved, drain workloads and restart the host."
            ),
        )
        return record, action

    # Replace the original scenario result with the post-repair scan while preserving before/after evidence.
    after_probe = probe_memory_capacity(context.hardware, context.managed_memory_mb)
    after_record = evaluate_memory_scan(rule, {"memory_scan": after_probe})
    action["after_evidence"] = after_probe
    if after_record["status"] == "PASS":
        action.update(
            status="SUCCESS",
            message=(
                "Global capacity passed after Drop Cache, "
                f"matched granularity: {after_probe.get('matched_granularity_mb')}MB."
            ),
        )
    elif after_probe.get("failure_kind") == "capacity":
        action.update(
            status="NO_EFFECT",
            message=(
                "Drop Cache ran, but the A3 1GB/2MB rescan still reports insufficient global capacity. "
                "Drain workloads, restart the host, and rerun the check."
            ),
        )
    else:
        action.update(
            status="FAILED",
            message=(
                "Drop Cache ran, but the rescan did not produce valid capacity evidence. "
                "Restore scan capability first. If this cannot be resolved, drain workloads and restart the host."
            ),
        )
    return after_record, action


def fix_a5_hugepages(
    rule: dict[str, Any], context: CheckContext, record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Clear configured 950PR&950DT Products huge pages when explicitly authorized."""
    residual = record.get("evidence", {}).get("residual_entries", [])
    action: dict[str, Any] = {
        "id": record["id"],
        "type": "clear_a5_hugepages",
        "status": "SKIPPED",
        "message": "",
        "before_evidence": record.get("evidence", {}),
        "operation": {"paths": [entry["path"] for entry in residual], "value": "0"},
        "after_evidence": None,
    }
    if not residual:
        action.update(status="SUCCESS", message="No residual 950PR&950DT Products huge pages require cleanup.")
        return record, action
    if os.geteuid() != 0:
        action.update(status="FAILED", message="950PR&950DT Products huge-page cleanup requires root in the container.")
        return record, action

    failures = []
    for entry in residual:
        try:
            Path(entry["path"]).write_text("0\n", encoding="ascii")
        except OSError as exc:
            failures.append(f"{entry['path']}: {exc}")
    after_probe = probe_a5_hugepages()
    after_record = evaluate_a5_hugepages(rule, {"a5_hugepages": after_probe})
    action["after_evidence"] = after_probe
    if failures:
        action.update(status="FAILED", message="; ".join(failures))
    elif after_record["status"] == "PASS":
        action.update(status="SUCCESS", message="All 950PR&950DT Products 1GB and 2MB huge pages were cleared.")
    else:
        action.update(status="NO_EFFECT", message="Residual 950PR&950DT Products huge pages remain after cleanup.")
    return after_record, action


def execute_fix(
    catalog: dict[str, Any], context: CheckContext, check_results: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute authorized local repairs and return final scenario and action records."""
    rules = {rule["id"]: rule for rule in catalog["rules"]}
    final_results = list(check_results)
    actions: list[dict[str, Any]] = []
    progress = context.progress or ProgressReporter(False)
    for index, record in enumerate(check_results):
        if record["status"] not in ("BLOCK", "MANUAL_REQUIRED", "WARN"):
            continue
        # Execute only explicitly supported remediations; record all others as MANUAL.
        remediation = record["remediation"]
        with progress.stage(f"Repair {record['id']}"):
            if remediation["type"] == "drop_caches_then_rescan":
                final_results[index], action = fix_memory_with_drop_caches(rules[record["id"]], context, record)
                actions.append(action)
            elif remediation["type"] == "clear_a5_hugepages":
                final_results[index], action = fix_a5_hugepages(rules[record["id"]], context, record)
                actions.append(action)
                memory_index = next(
                    (idx for idx, item in enumerate(final_results) if item["id"] == "MEM-A5-001"),
                    None,
                )
                if memory_index is not None:
                    memory_probe = probe_a5_memory_capacity(context.managed_memory_mb)
                    final_results[memory_index] = evaluate_a5_memory(rules["MEM-A5-001"], {"a5_memory": memory_probe})
            else:
                actions.append(
                    {
                        "id": record["id"],
                        "type": remediation["type"],
                        "status": "MANUAL",
                        "message": remediation["description"],
                    }
                )
    return final_results, actions


def build_repair_summary(
    requested: bool, actions: list[dict[str, Any]], before_summary: dict[str, Any] | None
) -> dict[str, Any]:
    statuses = ("SUCCESS", "NO_EFFECT", "FAILED", "SKIPPED", "MANUAL")
    return {
        "requested": requested,
        "before_summary": before_summary if requested else None,
        "action_count": len(actions),
        "status_counts": {status: sum(1 for action in actions if action["status"] == status) for status in statuses},
        "actions": actions,
    }


def result_exit_code(check_results: list[dict[str, Any]]) -> int:
    if any(is_blocking_result(result) for result in check_results):
        return 20
    if any(result["status"] in ("WARN", "UNKNOWN") for result in check_results):
        return 10
    return 0


def build_summary(check_results: list[dict[str, Any]], exit_code: int) -> dict[str, Any]:
    """Build overall status, counts, and blocker summaries from scenario results."""
    blockers = [result for result in check_results if is_blocking_result(result)]
    automatic_blockers = [result for result in blockers if result["status"] == "BLOCK"]
    warnings = [result for result in check_results if result["status"] == "WARN"]
    return {
        "overall_status": "BLOCK" if exit_code == 20 else ("WARN" if exit_code == 10 else "PASS"),
        "blocking_count": len(blockers),
        "automatic_block_count": len(automatic_blockers),
        "blocking_not_executed_count": sum(1 for result in blockers if result["status"] == "NOT_EXECUTED"),
        "warning_count": len(warnings),
        "pass_count": sum(1 for result in check_results if result["status"] == "PASS"),
        "not_applicable_count": sum(1 for result in check_results if result["status"] == "NOT_APPLICABLE"),
        "not_executed_count": sum(1 for result in check_results if result["status"] == "NOT_EXECUTED"),
        "manual_required_count": sum(1 for result in check_results if result["status"] == "MANUAL_REQUIRED"),
        "blockers": [
            {"id": result["id"], "problem": result["problem"], "detail": result["detail"]} for result in blockers
        ],
        "warnings": [
            {"id": result["id"], "problem": result["problem"], "detail": result["detail"]} for result in warnings
        ],
    }


def format_argument_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_text(result: dict[str, Any], canonical_result_path: Path | None = None) -> str:
    """Render structured check results as a human-readable TEXT report."""
    statuses = ("BLOCK", "MANUAL_REQUIRED", "WARN", "PASS", "UNKNOWN", "NOT_EXECUTED", "NOT_APPLICABLE")
    counts = {status: sum(1 for record in result["check_results"] if record["status"] == status) for status in statuses}
    summary = result["summary"]
    arguments = result["invocation"]["arguments"]
    repair = result["repair"]
    # Show the timestamp and effective invocation arguments at the top of the report.
    lines = [
        f"KV Pool Environment Check Report v{result['tool_version']} | run_id={result['run_id']}",
        "=" * 72,
        f"Check time: {result['timestamp_iso']}",
        "Invocation arguments:",
        *(f"- {key}: {format_argument_value(value)}" for key, value in arguments.items()),
        "",
        f"Final result: {summary['overall_status']} | blockers={summary['blocking_count']}"
        f" (automatic={summary['automatic_block_count']}, manual={summary['manual_required_count']}, "
        f"not executed={summary['blocking_not_executed_count']}) warnings={summary['warning_count']} "
        f"passed={summary['pass_count']} not applicable={summary['not_applicable_count']}",
        "Status counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()),
    ]
    if repair["requested"]:
        repair_counts = repair["status_counts"]
        lines.append(
            "Repair results: "
            + ", ".join(
                f"{status}={repair_counts[status]}"
                for status in ("SUCCESS", "NO_EFFECT", "FAILED", "SKIPPED", "MANUAL")
            )
        )
        if repair["actions"]:
            lines.extend(["", "Repair actions:"])
            lines.extend(f"- {item['id']} {item['status']}: {item['message']}" for item in repair["actions"])
    # Show remediation recommendations only when --fix was not requested, avoiding duplicate output.
    blockers = [record for record in result["check_results"] if is_blocking_result(record)]
    warnings = [record for record in result["check_results"] if record["status"] == "WARN"]
    if blockers and not repair["requested"]:
        lines.extend(["", "Blocker remediation:"])
        lines.extend(f"- {record['id']}: {record['remediation']['description']}" for record in blockers)
    if warnings and not repair["requested"]:
        lines.extend(["", "Recommended confirmations:"])
        lines.extend(f"- {record['id']}: {record['remediation']['description']}" for record in warnings)
    for record in result["check_results"]:
        applicability = record["applicability"]
        conclusions = {
            "PASS": "Issue not detected",
            "BLOCK": "Issue detected",
            "WARN": "Risk detected; confirmation required",
            "MANUAL_REQUIRED": "Manual confirmation required",
            "NOT_EXECUTED": "Not executed in the current environment",
            "NOT_APPLICABLE": "Not applicable to the current deployment mode",
            "UNKNOWN": "Cannot confirm",
        }
        lines.extend(
            [
                "",
                f"[{record['status']}] {record['id']}",
                f"Scenario: {record['problem']}",
                f"Conclusion: {conclusions.get(record['status'], 'Cannot confirm')}",
            ]
        )
        positioning = record.get("positioning")
        if positioning:
            lines.extend(
                [
                    f"Layer: {positioning['layer']}",
                    f"Target: {positioning['target']}",
                    f"Coverage: {positioning['coverage']}",
                    f"Excludes: {'; '.join(positioning['excludes'])}",
                ]
            )
        if record["status"] != "PASS":
            lines.append(f"Impact: {record['impact']}")
        lines.extend(
            [
                f"Details: {record['detail']}",
                f"Evidence source: {record['evidence_method']}",
                f"Applicable hardware: {','.join(applicability['hardware'])} | "
                f"applicable backends: {','.join(applicability['backends'])}",
                f"Currently blocking: {'yes' if is_blocking_result(record) else 'no'}",
            ]
        )
    references = result.get("reference_documents", [])
    if references:
        lines.extend(["", "References:"])
        lines.extend(f"- {item['title']}: {item['url']}" for item in references)
    if canonical_result_path is None:
        lines.extend(
            [
                "",
                "Output note: only the report is printed; use --output-dir to save both "
                "JSON results and the TEXT report.",
            ]
        )
    else:
        lines.extend(["", f"Complete result (JSON): {canonical_result_path}"])
    return "\n".join(lines) + "\n"


def write_output_bundle(output_dir: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    """Write complete JSON results and the TEXT report with the same run_id."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    prefix = f"kvpool_envcheck_{result['run_id']}"
    result_path = output_dir / f"{prefix}.result.json"
    report_path = output_dir / f"{prefix}.report.txt"
    if result_path.exists() or report_path.exists():
        raise OSError(f"Output files already exist: {prefix}")

    json_tmp = output_dir / f".{result_path.name}.tmp"
    report_tmp = output_dir / f".{report_path.name}.tmp"
    result_committed = False
    try:
        write_utf8(json_tmp, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        write_utf8(report_tmp, render_text(result, result_path))
        os.replace(json_tmp, result_path)
        result_committed = True
        os.replace(report_tmp, report_path)
    except Exception:
        if result_committed:
            result_path.unlink(missing_ok=True)
        raise
    finally:
        json_tmp.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)
    return result_path, report_path


def render_completion(result: dict[str, Any], result_path: Path, report_path: Path) -> str:
    return (
        "\n".join(
            [
                "Check complete",
                f"Final result: {result['summary']['overall_status']}",
                f"run_id: {result['run_id']}",
                f"Complete result (JSON): {result_path}",
                f"Report (TEXT): {report_path}",
            ]
        )
        + "\n"
    )


def panorama(catalog: dict[str, Any]) -> str:
    lines = [
        "| Issue ID | Problem | Scope | Impact | Evidence source | Evaluation rule | Remediation | "
        "Blocking | Hardware | Backend | Link | Deployment | Runtime scope | Implementation |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rule in catalog["rules"]:
        params = json.dumps(rule["parameters"], ensure_ascii=False, separators=(",", ":"))
        positioning = rule.get("positioning")
        position_text = (
            "-"
            if not positioning
            else (
                f"{positioning['layer']}; {positioning['target']}; {positioning['coverage']}; "
                f"excludes: {', '.join(positioning['excludes'])}"
            )
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rule["id"]),
                    str(rule["problem"]),
                    position_text,
                    str(rule["impact"]),
                    str(rule["evidence_method"]),
                    f"`{rule['evaluator']}` `{params}`",
                    str(rule["remediation"]["description"]),
                    "yes" if rule["blocking"] else "no",
                    ",".join(rule["hardware"]),
                    ",".join(rule["backends"]),
                    ",".join(rule.get("link_types", ["all"])),
                    ",".join(rule.get("deployment_modes", ["all"])),
                    str(rule["scope"]),
                    "implemented",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_panorama(path: Path, catalog: dict[str, Any]) -> None:
    content = read_utf8(path) if path.exists() else "# KV Pool Check Panorama\n\n"
    begin = "<!-- BEGIN GENERATED RULE PANORAMA -->"
    end = "<!-- END GENERATED RULE PANORAMA -->"
    generated = begin + "\n\n" + panorama(catalog) + "\n\n" + end
    if begin in content and end in content:
        content = content.split(begin, 1)[0] + generated + content.split(end, 1)[1]
    else:
        content = content.rstrip() + "\n\n" + generated + "\n"
    write_utf8(path, content)


def load_catalog(path: Path) -> dict[str, Any]:
    """Read and validate rules.json, returning a catalog usable for checks."""
    try:
        catalog = json.loads(read_utf8(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Failed to read rule catalog: {exc}")
    validate_catalog(catalog)
    return catalog


def build_parser(catalog_data: dict[str, Any]) -> argparse.ArgumentParser:
    """Build detection, output, repair, and catalog-management arguments from the catalog."""
    hardware_metavar = "{" + ",".join(catalog_data["supported_hardware"]) + "}"
    backend_metavar = "{" + ",".join(catalog_data["supported_backends"]) + "}"
    link_type_metavar = "{" + ",".join(catalog_data["supported_link_types"]) + "}"
    deployment_mode_metavar = "{" + ",".join(catalog_data["supported_deployment_modes"]) + "}"
    parser = argparse.ArgumentParser(
        description=(
            f"vLLM-Ascend KV Pool offline deployment environment checker v{TOOL_VERSION}\n"
            "Runs only inside a Docker container, operates offline, and does not modify the environment by default."
        ),
        epilog=(
            "Examples:\n"
            "  Print the check report:\n"
            "    python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck "
            "--hardware A3 --backend memcache --managed-memory-mb 32768 "
            "--deployment-mode multi-node --link-type hccs\n\n"
            "  Save complete JSON results and the TEXT report:\n"
            "    python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck "
            "--hardware A3 --backend memcache --managed-memory-mb 32768 "
            "--deployment-mode multi-node --link-type hccs --output-dir ./results\n\n"
            "  Validate a 950PR&950DT Products multi-node deployment with a remote UB EID:\n"
            "    python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck "
            "--hardware A5 --backend mooncake --managed-memory-mb 32768 "
            "--deployment-mode multi-node --link-type ub --peer-eid <remote-eid>\n\n"
            "  Authorize supported local repairs:\n"
            "    python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck "
            "--hardware A3 --backend memcache --managed-memory-mb 32768 "
            "--deployment-mode multi-node --link-type hccs --fix --output-dir ./results\n\n"
            "  Validate the rule catalog:\n"
            "    python -m vllm_ascend.distributed.kv_transfer.utils.envcheck.kvpool_envcheck "
            "--validate-catalog\n\n"
            "Exit codes: 0=passed  10=warnings  20=blocking issues  2=input, rule, or output error"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    general = parser.add_argument_group("General arguments")
    general.add_argument("-h", "--help", action="help", help="Show help and exit")
    general.add_argument(
        "--catalog",
        type=Path,
        metavar="FILE",
        default=DEFAULT_CATALOG,
        help="Rule file (default: rules.json in the tool directory)",
    )

    detection = parser.add_argument_group("Check arguments (required in check mode)")
    detection.add_argument(
        "--hardware",
        metavar=hardware_metavar,
        help="Hardware platform; supported values come from the rule file and must match the detected hardware",
    )
    detection.add_argument(
        "--backend", metavar=backend_metavar, help="KV Pool backend; supported values come from the rule file"
    )
    detection.add_argument(
        "--managed-memory-mb",
        type=int,
        metavar="MB",
        help="Total backend-managed memory in MB; must be greater than 0 (example: 32768)",
    )
    detection.add_argument(
        "--deployment-mode",
        metavar=deployment_mode_metavar,
        help="Deployment scale: single-node or multi-node; selects deployment-specific rules",
    )
    detection.add_argument(
        "--link-type",
        metavar=link_type_metavar,
        help="Backend communication link; A2 uses roce, A3 uses roce or hccs, and 950PR&950DT Products use ub",
    )
    detection.add_argument(
        "--peer-eid",
        metavar="EID",
        help="Remote 950PR&950DT Products UB EID for optional multi-node urma_ping validation",
    )

    ssd = parser.add_argument_group("SSD Offload (optional)")
    ssd.add_argument("--ssd-offload", action="store_true", help="Enable SSD Offload dependency checks")
    ssd.add_argument("--ssd-path", metavar="PATH", help="SSD path; can be used only with --ssd-offload")

    output = parser.add_argument_group("Output arguments (optional)")
    output.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Directory for complete JSON results and the TEXT report; print only the report when omitted",
    )
    output.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="Progress output: auto=interactive terminal only (default), always=always, never=disabled",
    )

    repair = parser.add_argument_group("Repair flow (optional)")
    repair.add_argument(
        "--fix",
        action="store_true",
        help="Authorize and execute supported local repairs; mark all other repairs as MANUAL",
    )

    catalog = parser.add_argument_group("Rule management (no check arguments required)")
    catalog.add_argument("--validate-catalog", action="store_true", help="Validate the rule file and exit")
    catalog.add_argument("--list-rules", action="store_true", help="List all scenarios and exit")
    catalog.add_argument("--describe-rule", metavar="ID", help="Print the complete rule for one scenario")
    catalog.add_argument(
        "--sync-panorama",
        type=Path,
        metavar="DESIGN_FILE",
        help=(
            "After modifying rules.json, synchronize the rule panorama into the generated section of a design document"
        ),
    )
    return parser


def preparse_catalog_path(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args, _unknown = parser.parse_known_args(argv)
    return args.catalog


def load_cli_catalog(argv: list[str]) -> tuple[dict[str, Any], argparse.ArgumentParser]:
    """Load the requested rule catalog and build a complete parser for it."""
    catalog_path = preparse_catalog_path(argv)
    catalog = load_catalog(catalog_path)
    return catalog, build_parser(catalog)


def run_catalog_command(args: argparse.Namespace, catalog: dict[str, Any]) -> int | None:
    """Run a catalog-management command, or return None when none was requested."""
    if args.validate_catalog:
        print(f"catalog valid: schema={catalog['schema_version']} rules={len(catalog['rules'])}")
        return 0
    if args.list_rules:
        for rule in catalog["rules"]:
            print(f"{rule['id']}\t{rule['problem']}\t{'BLOCK' if rule['blocking'] else 'WARN/PASS'}")
        return 0
    if args.describe_rule:
        rule = next((item for item in catalog["rules"] if item["id"] == args.describe_rule), None)
        if rule is None:
            raise InputError(f"Unknown scenario ID: {args.describe_rule}")
        print(json.dumps(rule, ensure_ascii=False, indent=2))
        return 0
    if args.sync_panorama:
        write_panorama(args.sync_panorama, catalog)
        print(f"Rule panorama synchronized: {args.sync_panorama}")
        return 0
    return None


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run catalog management or the environment check."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        # Load the catalog and build a parser whose supported values come from it.
        catalog, parser = load_cli_catalog(raw_argv)
        args = parser.parse_args(raw_argv)

        # Catalog query, validation, and document-sync commands return before the environment check.
        catalog_exit_code = run_catalog_command(args, catalog)
        if catalog_exit_code is not None:
            return catalog_exit_code

        # Validate runtime location, required arguments, feature arguments, and hardware consistency.
        progress = ProgressReporter.from_mode(args.progress)
        effective_progress = "always" if progress.enabled else "never"
        with progress.stage("Input and hardware validation"):
            if not args.hardware or not args.backend or not args.deployment_mode or not args.link_type:
                raise InputError("Check mode requires --hardware, --backend, --deployment-mode, and --link-type")
            if not is_docker_runtime():
                raise InputError("The tool supports execution only inside a Docker container")
            if not args.managed_memory_mb or args.managed_memory_mb <= 0:
                raise InputError("Check mode requires --managed-memory-mb to be greater than 0")
            if args.ssd_path and not args.ssd_offload:
                raise InputError("--ssd-path can be used only with --ssd-offload")
            if args.peer_eid and (
                args.hardware.strip().upper() != "A5"
                or args.link_type.strip().lower() != "ub"
                or args.deployment_mode.strip().lower() != "multi-node"
            ):
                raise InputError("--peer-eid requires --hardware A5 --link-type ub --deployment-mode multi-node")
            hardware, backend, link_type, deployment_mode, _hardware_identification = validate_inputs(
                catalog, args.hardware, args.backend, args.link_type, args.deployment_mode
            )

        # Run applicable probes and build the pre-repair summary.
        context = CheckContext(
            hardware=hardware,
            backend=backend,
            managed_memory_mb=args.managed_memory_mb,
            deployment_mode=deployment_mode,
            link_type=link_type,
            ssd_offload=args.ssd_offload,
            ssd_path=args.ssd_path,
            peer_eid=args.peer_eid,
            progress=progress,
        )
        check_results = collect_check_results(catalog, context)
        before_exit_code = result_exit_code(check_results)
        before_summary = build_summary(check_results, before_exit_code)

        # Execute authorized repairs only when --fix is explicit, then use post-repair results.
        if args.fix:
            check_results, actions = execute_fix(catalog, context, check_results)
        else:
            actions = []

        # Build one result model containing inputs, summaries, scenarios, repairs, and references.
        exit_code = result_exit_code(check_results)
        now = datetime.datetime.now(datetime.timezone.utc)  # noqa: UP017
        timestamp = int(now.timestamp())
        run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(4)}"
        invocation = {
            "arguments": {
                "hardware": hardware,
                "backend": backend,
                "managed_memory_mb": args.managed_memory_mb,
                "deployment_mode": deployment_mode,
                "link_type": link_type,
                "ssd_offload": args.ssd_offload,
                "ssd_path": args.ssd_path,
                "peer_eid": args.peer_eid,
                "fix": args.fix,
                "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
                "catalog": str(args.catalog.resolve()),
                "progress": effective_progress,
            }
        }
        result = {
            "tool_version": TOOL_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": timestamp,
            "timestamp_iso": now.isoformat(),
            "invocation": invocation,
            "summary": build_summary(check_results, exit_code),
            "repair": build_repair_summary(args.fix, actions, before_summary),
            "check_results": check_results,
            "reference_documents": REFERENCE_DOCUMENTS,
            "exit_code": exit_code,
        }

        # Save JSON/TEXT when an output directory is provided; otherwise print only the TEXT report.
        if args.output_dir:
            result_path, report_path = write_output_bundle(args.output_dir, result)
            sys.stdout.write(render_completion(result, result_path, report_path))
        else:
            sys.stdout.write(render_text(result))
        return exit_code
    except (CatalogError, InputError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Check interrupted by the user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
