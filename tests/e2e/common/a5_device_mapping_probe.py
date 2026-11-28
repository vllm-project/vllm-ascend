# SPDX-License-Identifier: Apache-2.0
"""Model-free SDK probe coordinator; every selected device gets a fresh process."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEVICE_TIMEOUT_SECONDS = 60
TRANSPORT_TIMEOUT_SECONDS = 120
ENV_KEYS = (
    "ASCEND_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "LD_LIBRARY_PATH",
    "ASCEND_GLOBAL_RESOURCE_CONFIG",
    "ASCEND_LOCAL_COMM_RES",
    "VLLM_HOST_IP",
    "HCCL_IF_IP",
    "HCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "ASCEND_GLOBAL_LOG_LEVEL",
    "ASCEND_SLOG_PRINT_TO_STDOUT",
)
REQUIRED_APIS = {
    "aclInit",
    "aclrtGetDeviceCount",
    "aclrtSetDevice",
    "aclrtGetDevice",
    "aclrtGetLogicDevIdByUserDevId",
    "aclrtGetPhyDevIdByUserDevId",
    "aclrtGetUserDevIdByPhyDevId",
    "aclrtGetUserDevIdByLogicDevId",
    "aclrtResetDevice",
    "aclFinalize",
}


def validate_reports(reports, required_apis=REQUIRED_APIS):
    """Return all errors, including incomplete/crashed probes; never infer IDs."""
    errors = []
    physical_ids = []
    if len(reports) != 2 or {r["role"] for r in reports} != {"P", "D"}:
        errors.append("Expected one report for each of P and D")
    for report in reports:
        mask = report["env"].get("ASCEND_RT_VISIBLE_DEVICES") or ""
        assigned = mask.split(",")
        if len(assigned) != 2 or any(not value.strip().isdigit() for value in assigned):
            errors.append(f"{report['role']}: invalid two-device runtime mask {mask!r}")
            continue
        if [p["user"] for p in report["probes"]] != [0, 1]:
            errors.append(f"{report['role']}: incomplete device probes")
        for probe in report["probes"]:
            label = f"{report['role']} user={probe['user']}"
            if probe["returncode"] != 0 or probe.get("timeout"):
                errors.append(f"{label}: child failed or timed out")
            calls = [e for e in probe["events"] if e.get("event") == "return"]
            if len(calls) != len(required_apis) or {e["api"] for e in calls} != required_apis:
                errors.append(f"{label}: missing SDK results")
            for call in calls:
                if call["rc"] != 0:
                    errors.append(f"{label}: {call['api']} returned {call['rc']}")
            mappings = [e for e in probe["events"] if e.get("event") == "mapping"]
            if len(mappings) != 1:
                errors.append(f"{label}: missing mapping result")
                continue
            mapping = mappings[0]
            user = probe["user"]
            if user not in (0, 1):
                errors.append(f"{label}: unexpected user device ID")
                continue
            if mapping["count"] != 2:
                errors.append(f"{label}: expected two SDK-visible devices, got {mapping['count']}")
            if any(mapping[key] != user for key in ("user", "current", "user_from_physical", "user_from_logical")):
                errors.append(f"{label}: device selection or inverse mapping mismatch")
            if mapping["logical"] != int(assigned[user]):
                errors.append(f"{label}: runtime mask and SDK logical ID disagree")
            if mapping["physical"] < 0:
                errors.append(f"{label}: invalid physical ID")
            physical_ids.append(mapping["physical"])
    if len(physical_ids) != 4 or len(set(physical_ids)) != 4:
        errors.append(f"Expected four distinct physical devices, got {physical_ids}")
    return errors


def validate_transport_reports(reports):
    """Require all same-process snapshots and exactly one successful init."""
    errors = []
    snapshot_apis = REQUIRED_APIS - {"aclInit", "aclrtSetDevice", "aclrtResetDevice", "aclFinalize"}
    stages = ("after_torch", "after_import", "after_initialize")
    for report in reports:
        for probe in report["probes"]:
            events = probe["events"]
            transport = [e for e in events if e.get("event") == "transport"]
            starts = [e for e in transport if e["stage"] == "initialize_start"]
            returns = [e for e in transport if e["stage"] == "initialize_return"]
            if len(starts) != 1 or len(returns) != 1 or returns[0].get("rc") != 0:
                errors.append(f"{report['role']} user={probe['user']}: initialization missing or failed")
            if sum(e["stage"] == "complete" for e in transport) != 1:
                errors.append(f"{report['role']} user={probe['user']}: did not complete")
            mappings = [e for e in events if e.get("event") == "mapping"]
            if len(mappings) != len(stages) or any(m != mappings[0] for m in mappings):
                errors.append(f"{report['role']} user={probe['user']}: mapping changed or snapshot missing")
    for stage in stages:
        snapshots = []
        for report in reports:
            probes = []
            for probe in report["probes"]:
                events = []
                active = False
                for item in probe["events"]:
                    if item.get("event") == "transport":
                        if item["stage"] == stage + "_start":
                            active = True
                        elif item["stage"] == stage + "_return":
                            active = False
                    elif active:
                        events.append(item)
                probes.append({**probe, "events": events})
            snapshots.append({**report, "probes": probes})
        errors.extend(f"{stage}: {error}" for error in validate_reports(snapshots, snapshot_apis))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--role", required=True, choices=("P", "D"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--transport-library")
    args = parser.parse_args()
    report = {"role": args.role, "pid": os.getpid(), "env": {k: os.environ.get(k) for k in ENV_KEYS}, "probes": []}
    output = Path(args.output)
    # Preserve the effective child environment even if the native SDK hangs.
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for user in (0, 1):
        events: list[dict[str, object]] = []
        probe = {"user": user, "events": events, "returncode": None, "timeout": False}
        command = [args.binary, str(user)]
        timeout = DEVICE_TIMEOUT_SECONDS
        if args.transport_library:
            command = [
                sys.executable,
                str(Path(__file__).with_name("a5_mooncake_init_probe.py")),
                "--library",
                args.transport_library,
                "--user",
                str(user),
            ]
            timeout = TRANSPORT_TIMEOUT_SECONDS
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
            probe["pid"] = child.pid
            try:
                stdout, stderr = child.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                stdout, stderr = child.communicate()
                probe["timeout"] = True
            probe["returncode"] = child.returncode
        stdout_text = stdout.decode("utf-8", errors="replace")
        probe["stdout"] = stdout_text
        probe["stderr"] = stderr.decode("utf-8", errors="replace")
        for line in stdout_text.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and "event" in event:
                events.append(event)
        report["probes"].append(probe)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("A5_DEVICE_MAPPING " + json.dumps({**report, "probes": [probe]}), flush=True)
    # Validation belongs to the parent, which can compare the two physical sets.


if __name__ == "__main__":
    main()
