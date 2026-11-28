# SPDX-License-Identifier: Apache-2.0
"""Check SDK mappings under the real PD harness's 2+2 masks, without a model.

This is not a vLLM worker, HIXL connection, or PD serving success test.
"""

import json
import os
import shlex
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from tests.e2e.common.a5_device_mapping_probe import ENV_KEYS, validate_reports
from tests.e2e.conftest import RemotePDServer

COMMON = Path(__file__).resolve().parents[2] / "common"
ROLE_TIMEOUT_SECONDS = 150


def build_probe(output_dir):
    candidates = [
        os.environ.get("ASCEND_HOME_PATH"),
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/cann",
    ]
    sdk = next((Path(p) for p in candidates if p and (Path(p) / "include/acl/acl.h").is_file()), None)
    assert sdk is not None, f"Missing SDK headers; checked {candidates}"
    library = sdk / "lib64"
    assert (library / "libascendcl.so").exists(), f"Missing SDK library in {library}"
    binary = output_dir / "a5-device-mapping"
    command = [
        *shlex.split(os.environ.get("CXX", "c++")),
        "-std=c++17",
        str(COMMON / "a5_device_mapping_probe.cpp"),
        f"-I{sdk / 'include'}",
        f"-L{library}",
        "-lascendcl",
        "-o",
        str(binary),
    ]
    versions = {}
    for name in ("version.cfg", "version.info", "ascend_toolkit_install.info"):
        path = sdk / name
        if path.is_file():
            versions[str(path)] = path.read_text(encoding="utf-8", errors="replace")[:8192]
    print(
        "A5_DEVICE_MAPPING_BUILD "
        + json.dumps({"command": command, "sdk": str(sdk.resolve()), "sdk_versions": versions}),
        flush=True,
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    print(result.stdout, result.stderr, flush=True)
    assert result.returncode == 0, "SDK mapping probe did not compile/link; see compiler output"
    return binary


class MappingProbeServers(RemotePDServer):
    """Keep the real allocation and Popen environment merge; replace only payload."""

    def __init__(self, binary, output_dir):
        self.binary = binary
        self.output_dir = output_dir
        self.roles = []
        self.reports = []
        try:
            super().__init__(
                [
                    ["--port", "18000", "--tensor-parallel-size", "1", "--prefill-context-parallel-size", "2"],
                    ["--port", "18001", "--tensor-parallel-size", "1", "--data-parallel-size", "2"],
                ]
            )
        finally:
            # Only children created by this instance; also covers partial startup.
            self._terminate_server()

    def _start_server_with_prefix(self, server_cmd, env_dict, log_prefix):
        role = "P" if not self.roles else "D"
        self.roles.append(role)
        print(
            "A5_DEVICE_MAPPING_LAUNCH "
            + json.dumps(
                {"role": role, "harness_command": server_cmd, "assigned_mask": env_dict["ASCEND_RT_VISIBLE_DEVICES"]}
            ),
            flush=True,
        )
        command = [
            sys.executable,
            str(COMMON / "a5_device_mapping_probe.py"),
            "--binary",
            str(self.binary),
            "--role",
            role,
            "--output",
            str(self.output_dir / f"{role}.json"),
        ]
        return super()._start_server_with_prefix(command, env_dict, log_prefix)

    def _wait_for_multiple_servers(self, *args, **kwargs):
        # No HTTP server or weights: wait for the short-lived SDK coordinators.
        failures = []
        for role, proc in zip(self.roles, self._proc_list):
            try:
                code = proc.wait(timeout=ROLE_TIMEOUT_SECONDS)
                if code != 0:
                    failures.append(f"{role}: coordinator exit {code}")
            except subprocess.TimeoutExpired:
                failures.append(f"{role}: coordinator timeout")
                self._terminate_process_tree(proc)
            path = self.output_dir / f"{role}.json"
            if not path.is_file():
                failures.append(f"{role}: missing report")
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            self.reports.append(report)
            # Existing selected-test logs archive this even on assertion failure.
            print("A5_DEVICE_MAPPING_REPORT " + json.dumps(report), flush=True)
        assert not failures, "\n".join(failures)


def test_a5_pd_device_mapping(tmp_path):
    packages = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "mooncake-transfer-engine-npu"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    print(
        "A5_DEVICE_MAPPING_PARENT "
        + json.dumps({"pid": os.getpid(), "env": {k: os.environ.get(k) for k in ENV_KEYS}, "packages": packages}),
        flush=True,
    )
    binary = build_probe(tmp_path)
    servers = MappingProbeServers(binary, tmp_path)
    errors = validate_reports(servers.reports)
    assert not errors, "\n".join(errors)
