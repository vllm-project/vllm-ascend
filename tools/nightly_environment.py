#!/usr/bin/env python3
"""Prepare isolated nightly roles using an explicitly limited baseline-image policy.

Server preserves its image's NPU/NumPy ABI stack; this is NOT a complete pip
dependency solution. Existing pip-check failures are archived and only unchanged
failures may remain. New failures or modified protected packages stop setup.
Client gets a clean venv and the fixed upstream AISBench API dependencies.
No server, benchmark, pytest, or NPU workload is launched by this module.
"""

import argparse
import importlib.metadata as metadata
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

AISBENCH_SHA = "0da56eadb2ac85c31c2540f4f5b69af3ec5717a5"
CANN_SCRIPTS = (
    "/usr/local/Ascend/cann/set_env.sh",
    "/usr/local/Ascend/ascend-toolkit/set_env.sh",
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh",
)
ATB_SCRIPT = "/usr/local/Ascend/nnal/atb/set_env.sh"
PROTECTED = (
    "torch",
    "torch-npu",
    "torchvision",
    "torchaudio",
    "numpy",
    "opencv-python-headless",
    "triton-ascend",
    "scipy",
    "numba",
)


def validate_sha(value):
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("A full lowercase frozen Git SHA is required")
    return value


def new_conflicts(before, after):
    def key(line):
        # Editable SHA builds change only the distribution owner version. Keep
        # the full requirement and actual dependency version in the comparison.
        return re.sub(r"^(\S+) \S+ (has requirement|requires|is not supported)", r"\1 <version> \2", line.strip())

    known = {key(line) for line in before.splitlines() if line.strip()}
    return [
        line
        for line in after.splitlines()
        if line.strip() and line.strip() != "No broken requirements found." and key(line) not in known
    ]


def check_baseline(before_versions, after_versions, before_check, after_check):
    changed = [name for name, version in before_versions.items() if after_versions.get(name) != version]
    failures = new_conflicts(before_check, after_check)
    if changed or failures:
        raise RuntimeError(f"Baseline changed: packages={changed}; new pip conflicts={failures}")


def web_requirements(text):
    lines = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    return [line for line in lines if re.match(r"^(fastapi|starlette|setuptools)\s*(?:\[|[<>=!~])", line)]


def validate_pip_check(code, text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if code == 0 and lines == ["No broken requirements found."]:
        return
    if (
        code == 1
        and lines
        and all(
            re.match(
                r"^\S+ \S+ (?:has requirement .+, but you have .+|requires .+, which is not installed|"
                r"is not supported on this platform)\.?$",
                line,
            )
            for line in lines
        )
    ):
        return
    raise RuntimeError(f"pip check failed outside standard dependency audit: exit={code}; {text}")


def activation(root, role):
    if role == "client":
        return (
            "unset PYTHONPATH PYTHONHOME\nexport PYTHONNOUSERSITE=1\n"
            f'export PATH={shlex.quote(str(root / "venv/bin"))}:"$PATH"\n'
        )
    return vendor_setup() + "nightly_load_environment || return 1\n"


def vendor_setup():
    candidates = " ".join(shlex.quote(path) for path in CANN_SCRIPTS)
    return (
        "nightly_load_environment() {\n"
        "  local cann_loaded=0 cann_env restore_nounset=0\n"
        '  case "$-" in *u*) restore_nounset=1;; esac\n'
        "  set +u\n"
        f"  for cann_env in {candidates}; do\n"
        '    if test -f "$cann_env"; then\n'
        '      source "$cann_env" >/dev/null || return 1\n'
        "      cann_loaded=1; break\n"
        "    fi\n  done\n"
        '  if test "$cann_loaded" != 1; then echo "CANN environment script missing" >&2; return 1; fi\n'
        f"  if test -f {shlex.quote(ATB_SCRIPT)}; then\n"
        f"    source {shlex.quote(ATB_SCRIPT)} >/dev/null || return 1\n"
        "  fi\n"
        '  export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"\n'
        '  if test "$restore_nounset" = 1; then set -u; fi\n'
        "}\n"
    )


def run(command, log_path, *, env=None, cwd=None, timeout=1800, check=True):
    started = time.monotonic()
    print("RUN", " ".join(map(str, command)), flush=True)
    with Path(log_path).open("w", encoding="utf-8") as output:
        result = subprocess.run(
            list(map(str, command)), stdout=output, stderr=subprocess.STDOUT, env=env, cwd=cwd, timeout=timeout
        )
    print(f"END exit={result.returncode} seconds={time.monotonic() - started:.1f} log={log_path}", flush=True)
    if check and result.returncode:
        print(Path(log_path).read_text(errors="replace")[-12000:], flush=True)
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")
    return result.returncode


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def clone_fixed(source, target, sha, root):
    validate_sha(sha)
    run(["git", "clone", "--no-hardlinks", source, target], root / (target.name + "-clone.log"), timeout=240)
    run(["git", "-C", target, "checkout", "--detach", sha], root / (target.name + "-checkout.log"), timeout=240)
    if git_head(target) != sha:
        raise RuntimeError(f"Checkout mismatch: {target}")


def versions():
    return {name: metadata.version(name) for name in PROTECTED}


def cann_environment():
    """Read only the installed vendor environment script, not arbitrary profiles."""
    environment = os.environ.copy()
    command = [
        "bash",
        "-c",
        vendor_setup() + "nightly_load_environment || exit 1\n"
        "exec \"$1\" -c 'import json,os;print(json.dumps(dict(os.environ)))'",
        "_",
        sys.executable,
    ]
    environment.update(json.loads(subprocess.check_output(command, text=True, timeout=30)))
    return environment


def prepare_server(args, root, source):
    environment = cann_environment()
    before = versions()
    (root / "protected-before.json").write_text(json.dumps(before, indent=2))
    before_path = root / "pip-check-before.log"
    before_code = run([sys.executable, "-m", "pip", "check"], before_path, env=environment, check=False, timeout=60)
    validate_pip_check(before_code, before_path.read_text())
    vllm = root / "vllm"
    ascend = root / "vllm-ascend"
    clone_fixed("https://github.com/vllm-project/vllm.git", vllm, args.vllm_sha, root)
    clone_fixed(str(source), ascend, git_head(source), root)
    run(
        ["git", "-C", ascend, "submodule", "update", "--init", "--recursive"],
        root / "ascend-submodule.log",
        env=environment,
        timeout=300,
    )
    constraints = root / "protected-constraints.txt"
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in before.items()))
    # Web-layer bounds come from the frozen vLLM source rather than a guessed
    # version. Do not resolve all requirements: Triton 3.2.2 pins NumPy 1.26.4,
    # while the same vLLM tree's OpenCV requirement needs NumPy >=2.
    common = (vllm / "requirements/common.txt").read_text()
    web = web_requirements(common)
    if not any(line.startswith("fastapi") for line in web):
        raise RuntimeError("Frozen vLLM web dependency declaration was not found")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--progress-bar",
            "off",
            "--timeout",
            "20",
            "--retries",
            "1",
            "-c",
            constraints,
            *web,
        ],
        root / "web-dependencies.log",
        env=environment,
        timeout=600,
    )
    vllm_env = dict(environment, VLLM_TARGET_DEVICE="empty")
    run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", vllm],
        root / "vllm-install.log",
        env=vllm_env,
        timeout=600,
    )
    run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", ascend],
        root / "ascend-install.log",
        env=environment,
        timeout=2400,
    )
    after = versions()
    after_path = root / "pip-check-after.log"
    after_code = run([sys.executable, "-m", "pip", "check"], after_path, env=environment, check=False, timeout=60)
    validate_pip_check(after_code, after_path.read_text())
    (root / "protected-after.json").write_text(json.dumps(after, indent=2))
    before_check = before_path.read_text()
    after_check = after_path.read_text()
    report = {
        "policy": "baseline-image-no-deps",
        "complete_dependency_solution": False,
        "source_sha": git_head(ascend),
        "vllm_sha": git_head(vllm),
        "pip_check_before": before_check,
        "pip_check_after": after_check,
        "new_conflicts": new_conflicts(before_check, after_check),
        "protected_before": before,
        "protected_after": after,
    }
    (root / "environment-report.json").write_text(json.dumps(report, indent=2))
    check_baseline(before, after, before_check, after_check)
    run(
        [
            sys.executable,
            "-c",
            "import vllm, vllm_ascend, torch, torch_npu; "
            "print(vllm.__version__); print(torch.__version__); print(vllm.__file__); "
            "print(vllm_ascend.__file__)",
        ],
        root / "imports.log",
        env=environment,
        timeout=120,
    )
    return sys.executable


def prepare_client(args, root, source):
    benchmark = root / "benchmark"
    clone_fixed(str(Path(args.benchmark_source).resolve(strict=True)), benchmark, AISBENCH_SHA, root)
    venv = root / "venv"
    run([sys.executable, "-m", "venv", venv], root / "venv.log", timeout=60)
    python = venv / "bin/python"
    environment = os.environ.copy()
    # CANN PYTHONPATH carries unrelated vendor distributions into an otherwise
    # clean venv. The HTTP client must not inherit those packages or a vLLM plugin.
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    constraints = root / "client-constraints.txt"
    constraints.write_text(
        "numpy==1.26.4\npandas==2.2.3\nopencv-python-headless==4.11.0.86\ntorch==2.10.0+cpu\ntransformers==5.14.1\n"
    )
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "--progress-bar",
            "off",
            "torch==2.10.0+cpu",
        ],
        root / "client-torch.log",
        env=environment,
        timeout=1200,
    )
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--progress-bar",
            "off",
            "-c",
            constraints,
            "-e",
            str(benchmark) + "[api]",
            "PyYAML",
        ],
        root / "client-install.log",
        env=environment,
        timeout=1800,
    )
    run([python, "-m", "pip", "check"], root / "pip-check-after.log", env=environment, timeout=60)
    run([venv / "bin/ais_bench", "--help"], root / "client-imports.log", env=environment, timeout=120)
    report = {
        "policy": "isolated-client-venv",
        "benchmark_sha": git_head(benchmark),
        "source_sha": git_head(source),
        "python": str(python),
        "pip_check": (root / "pip-check-after.log").read_text(),
    }
    (root / "environment-report.json").write_text(json.dumps(report, indent=2))
    return str(python)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("server", "client"), required=True)
    parser.add_argument("--vllm-sha", type=validate_sha, required=True)
    parser.add_argument("--dep-dir", type=Path, required=True)
    parser.add_argument("--benchmark-source", default="/vllm-workspace/vllm-ascend/benchmark")
    args = parser.parse_args()
    if not args.dep_dir.is_absolute() or ".." in args.dep_dir.parts:
        parser.error("--dep-dir must be a private absolute directory")
    source = Path(__file__).resolve().parents[1]
    verified = (source / ".github/vllm-main-verified.commit").read_text().strip()
    if verified != args.vllm_sha:
        parser.error("--vllm-sha does not match this frozen Ascend source")
    args.dep_dir.mkdir(parents=True, exist_ok=False)
    python = (prepare_server if args.role == "server" else prepare_client)(args, args.dep_dir, source)
    (args.dep_dir / "activate.sh").write_text(activation(args.dep_dir, args.role))
    print(
        json.dumps(
            {
                "role": args.role,
                "python": python,
                "dep_dir": str(args.dep_dir),
                "report": str(args.dep_dir / "environment-report.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
