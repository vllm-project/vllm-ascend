# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Phase 2 (max-seq, coarse ladder): 2k-4k-8k-16k-32k-64k probing per
(group, model) on the CI A5 (950) card, then ONE full-protocol round at the
max seq.

Probe = one short arm (steps 2 / warmup 2) at ascending ladder seqs until the
first failure (OOM shows as rc!=0 with no ARMRESULT); one half-step fallback if
even the first ladder rung fails. Full-protocol round mirrors the unified-table
format: delock 3 arms / activation 2 arms / saved 2 arms (pageable vs pinned),
all steps 8 / warmup 5, saved keeps credits 1 + manifest reuse from the probe.

CI adaptation of the 910B3 runner: paths are relative to this file, the model
ids and output dir come from env (PIN_M3 / PIN_M35 / PIN_OUT), and the ladder
is capped at 64k (65536) per protocol — do not probe beyond it.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DELOCK = f"{HERE}/delock_zero3"
ACT = f"{HERE}/activation_offload"
SAO = f"{HERE}/saved_activation_offload"
M3 = os.environ.get("PIN_M3", "Qwen/Qwen3-4B")
M35 = os.environ.get("PIN_M35", "Qwen/Qwen3.5-4B")
PY = sys.executable
OUT = os.environ.get("PIN_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)

# Ladder cap: 64k (65536) is the protocol maximum.
SEQ_64K = 65536

base_env = os.environ.copy()
base_env.update(
    {
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0"),
        "RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "LOCAL_RANK": "0",
    }
)
sao_env = base_env.copy()
sao_env["OMP_NUM_THREADS"] = str(os.cpu_count() or 42)  # matches source night-v2 protocol

BENCH_PATTERN = "zero3_offload_bench|activation_offload/bench|saved_activation_offload/bench"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def loadavg():
    with open("/proc/loadavg") as f:
        return " ".join(f.read().split()[:3])


def _read_jsonl(path):
    """Yield parsed JSON objects from path, skipping malformed lines."""
    with open(path) as f:
        for line in f:
            with contextlib.suppress(Exception):
                yield json.loads(line)


def _scan_arm_result(log_path):
    """Return the first ARMRESULT= JSON object in log_path, or None."""
    try:
        with open(log_path, "rb") as f:
            text = f.read().decode(errors="replace")
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith("ARMRESULT="):
            with contextlib.suppress(Exception):
                return json.loads(line[len("ARMRESULT=") :])
    return None


def wait_leftovers(max_s=900):
    waited = 0
    while waited < max_s:
        out = subprocess.run(["pgrep", "-f", BENCH_PATTERN], capture_output=True, text=True)
        pids = [p for p in out.stdout.split() if p.strip() and int(p) != os.getpid()]
        if not pids:
            return
        log(f"  waiting for leftover bench pids {pids} ({waited}s/{max_s}s)")
        time.sleep(15)
        waited += 15
    out = subprocess.run(["pgrep", "-f", BENCH_PATTERN], capture_output=True, text=True)
    for p in out.stdout.split():
        if p.strip():
            try:
                os.kill(int(p), signal.SIGKILL)
                log(f"  killed leftover pid {p}")
            except Exception:
                pass


def run_arm(script, args, log_path, env, timeout=1800, grace=30):
    with open(log_path, "w") as f:
        f.write(f"[arm cmd] {PY} {script} {' '.join(args)}\n")
        f.flush()
        p = subprocess.Popen([PY, script] + args, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    result = None
    deadline = time.time() + timeout
    seen_at = None
    while time.time() < deadline:
        rc = p.poll()
        if rc is not None:
            break
        if seen_at is None:
            result = _scan_arm_result(log_path)
            if result is not None:
                seen_at = time.time()
        elif time.time() - seen_at > grace:
            log(f"  grace expired after ARMRESULT, killing pid {p.pid}")
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait()
            break
        time.sleep(3)
    if seen_at is None:
        rc = p.poll()
        if rc is None:
            log(f"  TIMEOUT {timeout}s, killing pid {p.pid}")
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait()
            rc = p.poll()
        else:
            result = _scan_arm_result(log_path)
    tail = ""
    with contextlib.suppress(Exception), open(log_path, "rb") as f:
        tail = f.read().decode(errors="replace")[-1500:]
    return result, rc, tail


def arm_from_log(log_path):
    if not os.path.exists(log_path):
        return None
    return _scan_arm_result(log_path)


def probe(script, args, label, env, timeout):
    log_path = f"{OUT}/maxseq_probe_{label}.log"
    result, rc, tail = run_arm(script, args, log_path, env, timeout)
    if result is not None:
        step = result.get("step_avg_s") or result.get("step_median_s")
        log(f"  probe {label}: FITS (step~{step})")
        return True, result
    log(f"  probe {label}: FAILED rc={rc} (assumed OOM)")
    for line in tail.splitlines()[-10:]:
        log(f"    | {line}")
    return False, None


def probe_args(group, model, seq):
    if group == "delock":
        return [
            "--model",
            model,
            "--batch",
            "1",
            "--seq",
            str(seq),
            "--steps",
            "2",
            "--warmup",
            "2",
            "--pin",
            "0",
            "--register",
            "0",
        ]
    if group == "act":
        return [
            "--model",
            model,
            "--batch",
            "1",
            "--seq",
            str(seq),
            "--steps",
            "2",
            "--warmup",
            "2",
            "--zero-stage",
            "2",
            "--random-init",
            "--pin",
            "0",
        ]
    manifest = f"{OUT}/maxseq_{model.split('/')[-1]}_{seq}_manifest.json"
    return [
        "--model",
        model,
        "--batch",
        "1",
        "--seq",
        str(seq),
        "--offload-tensors",
        "3",
        "--offload-optimizer",
        "--act-mem-opt",
        "--steps",
        "2",
        "--warmup",
        "2",
        "--credits",
        "1",
        "--manifest-out",
        manifest,
        "--pin",
        "0",
    ]


def find_max(group, script, model, candidates, env, probe_timeout):
    short = model.split("/")[-1]
    last_ok = None
    for s in candidates:
        label = f"{group}_{short}_seq{s}"
        fits, _ = probe(script, probe_args(group, model, s), label, env, probe_timeout)
        if fits:
            last_ok = s
        else:
            break
    if last_ok is None and candidates[0] > 2048:
        s = candidates[0] // 2
        label = f"{group}_{short}_seq{s}"
        fits, _ = probe(script, probe_args(group, model, s), label, env, probe_timeout)
        if fits:
            last_ok = s
    return last_ok


def summarize(name, results):
    if name.startswith("delock"):
        un = results["unpinned"]
        pu = results["pinned-unregistered"]
        pr = results["pinned-registered"]
        if un is None or pu is None or pr is None:
            return {"experiment": name, "status": "failed"}
        return {
            "experiment": name,
            "unpinned_step_avg_s": un["step_avg_s"],
            "unpinned_step_min_s": un["step_min_s"],
            "pinned-unregistered_step_avg_s": pu["step_avg_s"],
            "pinned-registered_step_avg_s": pr["step_avg_s"],
            "pinned-registered_step_min_s": pr["step_min_s"],
            "speedup": un["step_avg_s"] / pr["step_avg_s"],
            "status": "ok",
        }
    if name.startswith("act"):
        pg, pin = results["pageable"], results["pinned"]
        if pg is None or pin is None:
            return {"experiment": name, "status": "failed"}
        return {
            "experiment": name,
            "pageable_step_avg_s": pg["step_avg_s"],
            "pageable_step_min_s": pg["step_min_s"],
            "pinned_step_avg_s": pin["step_avg_s"],
            "pinned_step_min_s": pin["step_min_s"],
            "speedup": pg["step_avg_s"] / pin["step_avg_s"],
            "status": "ok",
        }
    pg, pin = results["pageable"], results["pinned"]
    if pg is None or pin is None:
        return {"experiment": name, "status": "failed"}
    return {
        "experiment": name,
        "pageable_step_median_s": pg["step_median_s"],
        "pinned_step_median_s": pin["step_median_s"],
        "pin_speedup": pg["step_median_s"] / pin["step_median_s"],
        "status": "ok",
    }


def full_run(group, model, seq, timeout):
    short = model.split("/")[-1]
    name = f"{group}_maxseq_{short}"
    if group == "delock":
        script = (
            f"{DELOCK}/zero3_offload_bench.py" if "Qwen3-4B" in model else f"{DELOCK}/zero3_offload_bench_qwen35.py"
        )
        arms = [
            ("unpinned", ["--pin", "0", "--register", "0"]),
            ("pinned-unregistered", ["--pin", "1", "--register", "0"]),
            ("pinned-registered", ["--pin", "1", "--register", "1"]),
        ]
        common = ["--model", model, "--batch", "1", "--seq", str(seq), "--steps", "8", "--warmup", "5"]
        env = base_env
    elif group == "act":
        script = f"{ACT}/bench.py" if "Qwen3-4B" in model else f"{ACT}/bench_qwen35.py"
        arms = [("pageable", ["--pin", "0"]), ("pinned", ["--pin", "1"])]
        common = [
            "--model",
            model,
            "--batch",
            "1",
            "--seq",
            str(seq),
            "--steps",
            "8",
            "--warmup",
            "5",
            "--zero-stage",
            "2",
            "--random-init",
        ]
        env = base_env
    else:  # sao — two arms only (unified-table format), steps 8 / warmup 5
        script = f"{SAO}/bench.py"
        arms = [("pageable", ["--pin", "0"]), ("pinned", ["--pin", "1"])]
        common = [
            "--model",
            model,
            "--batch",
            "1",
            "--seq",
            str(seq),
            "--offload-tensors",
            "3",
            "--offload-optimizer",
            "--act-mem-opt",
            "--steps",
            "8",
            "--warmup",
            "5",
            "--credits",
            "1",
            "--manifest-out",
            f"{OUT}/maxseq_{short}_{seq}_manifest.json",
        ]
        env = sao_env
    results = {}
    for label, extra in arms:
        arm_log = f"{OUT}/{name}_seq{seq}_{label.replace(' ', '_')}.log"
        reused = arm_from_log(arm_log)
        if reused is not None:
            log(f"  arm {label}: reused from previous log")
            results[label] = reused
            continue
        log(f"  arm {label}: {' '.join(common + extra)}")
        result, rc, tail = run_arm(script, common + extra, arm_log, env, timeout)
        if result is None:
            log(f"  arm {label} FAILED rc={rc}")
            for line in tail.splitlines()[-12:]:
                log(f"    | {line}")
            results[label] = None
        else:
            results[label] = result
            key = result.get("step_avg_s") or result.get("step_median_s")
            log(f"  arm {label} done: step~{key}")
    summary = summarize(name, results)
    summary["max_seq"] = seq
    return summary


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    force_seq = None
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        force_seq = int(sys.argv[2])
    log(f"=== maxseq phase start (loadavg {loadavg()}) ===")
    wait_leftovers()
    results_file = f"{OUT}/results_maxseq.jsonl"
    done = set()
    if os.path.exists(results_file):
        for s in _read_jsonl(results_file):
            if s.get("status") == "ok":
                done.add((s["experiment"], s.get("max_seq")))

    plan = [
        ("delock", f"{DELOCK}/zero3_offload_bench.py", M3, [8192, 16384, 32768, SEQ_64K], base_env, 2400, 3600),
        (
            "delock",
            f"{DELOCK}/zero3_offload_bench_qwen35.py",
            M35,
            [4096, 8192, 16384, 32768, SEQ_64K],
            base_env,
            3600,
            3600,
        ),
        ("act", f"{ACT}/bench.py", M3, [8192, 16384, 32768, SEQ_64K], base_env, 2400, 3600),
        ("act", f"{ACT}/bench_qwen35.py", M35, [8192, 16384, 32768, SEQ_64K], base_env, 3600, 9000),
        ("sao", f"{SAO}/bench.py", M3, [8192, 16384, 32768, SEQ_64K], sao_env, 3600, 5400),
        ("sao", f"{SAO}/bench.py", M35, [4096, 8192, 16384, 32768, SEQ_64K], sao_env, 5400, 5400),
    ]
    if only:
        matched = []
        for p in plan:
            group, model = p[0], p[2]
            short = model.split("/")[-1]
            if only in (group, short, f"{group}_{short}"):
                matched.append(p)
        plan = matched
        if not plan:
            log(f"unknown group filter: {only}")
            return
    for group, script, model, candidates, env, probe_timeout, full_timeout in plan:
        short = model.split("/")[-1]
        log(f"=== maxseq {group} {short} start (loadavg {loadavg()}) ===")
        if force_seq:
            max_seq = force_seq
            log(f"  forced seq {max_seq} (skip probe) for {group} {short}")
        else:
            max_seq = find_max(group, script, model, candidates, env, probe_timeout)
        if max_seq is None:
            log(f"  NO seq fit for {group} {short}")
            with open(results_file, "a") as f:
                f.write(json.dumps({"experiment": f"{group}_maxseq_{short}", "status": "no-seq-fits"}) + "\n")
            continue
        log(f"  MAX SEQ for {group} {short}: {max_seq} — running full protocol (unified-table format)")
        if (f"{group}_maxseq_{short}", max_seq) in done:
            log(f"  already completed at seq {max_seq}, skip")
            continue
        summary = full_run(group, model, max_seq, full_timeout)
        with open(results_file, "a") as f:
            f.write(json.dumps(summary) + "\n")
        log(f"  SUMMARY: {json.dumps(summary)}")
    log("=== MAXSEQ PHASE DONE ===")


if __name__ == "__main__":
    main()
