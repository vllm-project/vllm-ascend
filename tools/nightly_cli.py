# SPDX-License-Identifier: Apache-2.0
"""Explicit preparation/verification for externally managed nightly processes."""

import argparse
import ast
import csv
import hashlib
import json
import re
import shlex
import shutil
import sys
from pathlib import Path

import yaml

from tools.aisbench_config import render_dataset_config, render_request_config, verify_performance


def select_case(path, case_name, benchmark):
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cases = [case for case in document["test_cases"] if case["name"] == case_name]
    if len(cases) != 1:
        raise ValueError("Select exactly one named test case")
    case = cases[0]
    config = case["benchmarks"][benchmark]
    if config["case_type"] != "performance":
        raise ValueError("Only performance benchmarks support this explicit CLI")
    return case, config


def template(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Template path escapes benchmark-home")
    return path.read_text(encoding="utf-8")


def write_private(directory, name, content):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(content)
    return str(destination)


def prepare(args, case, config):
    output = Path(args.output_dir).resolve()
    home = Path(args.benchmark_home).resolve()
    if output.is_relative_to(home):
        raise ValueError("output-dir must be private and outside benchmark-home")
    base = "ais_bench/benchmark/configs/"
    model = template(home, base + "models/vllm_api/" + config["request_conf"] + ".py")
    dataset = template(home, base + "datasets/" + config["dataset_conf"] + ".py")
    summarizer = template(home, base + "summarizers/perf/default_perf.py")
    options = dict(
        config,
        model=case["model"],
        model_path=args.model_path,
        host_ip=args.host,
        port=args.port,
        task_type=config["case_type"],
    )
    model = render_request_config(model, options)
    dataset_path = args.dataset_path
    dataset_digest = None
    if Path(dataset_path).is_file():
        if config["dataset_conf"].split("/")[0] != "gsm8k":
            raise ValueError("Single-file input is supported only for the GSM8K directory loader")
        directory = output / "dataset"
        directory.mkdir(parents=True, exist_ok=False)
        with Path(dataset_path).open("rb") as source, (directory / "train.jsonl").open("xb") as destination:
            shutil.copyfileobj(source, destination)
        shutil.copyfile(directory / "train.jsonl", directory / "test.jsonl")
        with (directory / "train.jsonl").open("rb") as source:
            dataset_digest = hashlib.file_digest(source, "sha256").hexdigest()
        dataset_path = str(directory)
    if config["dataset_conf"].startswith("textvqa"):
        dataset_path = str(Path(dataset_path) / "textvqa_val.jsonl")
    dataset = render_dataset_config(dataset, dataset_path)
    tree = ast.parse(dataset)
    names = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and (target.id == "datasets" or target.id.endswith("_datasets"))
    ]
    selected = "datasets" if "datasets" in names else names[0] if len(names) == 1 else None
    if selected is None:
        raise ValueError("Dataset template must expose one datasets list")
    content = model + "\n" + dataset + "\ndatasets = " + selected + "\n" + summarizer
    ast.parse(content)
    configuration = write_private(output, "benchmark.py", content)
    argv = ["ais_bench", configuration, "--mode", "perf"]
    if config.get("num_prompts"):
        argv += ["--num-prompts", str(config["num_prompts"])]
    argv += ["--work-dir", str(output / "results"), "--debug"]
    result = {
        "case": args.case,
        "benchmark": args.benchmark,
        "argv": argv,
        "config": configuration,
        "config_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "work_dir": str(output / "results"),
        "dataset_input_sha256": dataset_digest,
        "result_directory_evidence": "Performance Result files located in <directory>",
        "result_json_name": config["dataset_conf"].split("/")[0] + ".json",
        "result_csv_name": config["dataset_conf"].split("/")[0] + ".csv",
    }
    result.update(provenance(args, config))
    write_private(output, "benchmark.sh", "#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(argv) + "\n")
    write_private(output, "manifest.json", json.dumps(result, indent=2) + "\n")
    return result


def provenance(args, config):
    return {
        "source_config": str(Path(args.config).resolve()),
        "source_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "baseline": config.get("baseline", 1),
        "threshold": config.get("threshold", 0.97),
    }


def server_command(args, case, config):
    if case.get("service_mode", "openai") not in ("openai", "standard") or case.get("kv_pool"):
        raise ValueError("server-command supports only one directly managed OpenAI server")
    environment = {key: str(value) for key, value in case.get("envs", {}).items()}
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or "\0" in value for key, value in environment.items()):
        raise ValueError("Invalid YAML environment")
    environment["SERVER_PORT"] = str(args.port)

    def expand(value):
        value = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
            lambda match: environment[match[1] or match[2]],
            value,
        )
        if "$" in value or "\0" in value:
            raise ValueError("Unresolved or unsupported server argument expansion")
        return value

    arguments = [expand(value) for value in case["server_cmd"]]
    for flag, value in (("--port", str(args.port)), ("--host", args.host), ("--served-model-name", case["model"])):
        while flag in arguments:
            index = arguments.index(flag)
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError("Server flag has no value: " + flag)
            del arguments[index : index + 2]
        arguments = [argument for argument in arguments if not argument.startswith(flag + "=")]
        arguments += [flag, value]
    argv = ["vllm", "serve", args.model_path] + arguments
    script = "#!/usr/bin/env bash\nset -euo pipefail\n"
    script += "".join("export " + key + "=" + shlex.quote(value) + "\n" for key, value in environment.items())
    script += "exec " + shlex.join(argv) + "\n"
    result = {"case": args.case, "argv": argv, "environment": environment}
    result.update(provenance(args, config))
    result["script"] = write_private(args.output_dir, "server.sh", script)
    write_private(args.output_dir, "server.json", json.dumps(result, indent=2) + "\n")
    return result


def verify(args, config):
    result_json = Path(args.result_json).resolve()
    result_csv = Path(args.result_csv).resolve()
    data = json.loads(result_json.read_text(encoding="utf-8"))
    with result_csv.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        rows = list(reader)
    tpot = None
    if config.get("tpot_threshold") is not None:
        values = [row[header.index("Average")] for row in rows if row and row[0] == "TPOT"]
        if len(values) != 1:
            raise ValueError("Results must contain one TPOT row")
        tpot = values[0]
    result = {
        "case": args.case,
        "benchmark": args.benchmark,
        "result_json": str(result_json),
        "result_csv": str(result_csv),
        "verdict": "passed",
    }
    result.update(provenance(args, config))
    try:
        verify_performance(
            data,
            config.get("baseline", 1),
            config.get("threshold", 0.97),
            input_throughput_threshold=config.get("input_throughput_threshold"),
            tpot_threshold=config.get("tpot_threshold"),
            tpot=tpot,
        )
    except AssertionError as error:
        result.update(verdict="failed", reason=str(error))
    if args.output_file:
        output = Path(args.output_file)
        write_private(output.parent, output.name, json.dumps(result, indent=2) + "\n")
    return result


def locate_results(args, config):
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = provenance(args, config)
    if (
        manifest["case"] != args.case
        or manifest["benchmark"] != args.benchmark
        or any(manifest[key] != expected[key] for key in ("source_sha256", "benchmark_sha256"))
    ):
        raise ValueError("Manifest does not match the selected source and benchmark")
    prepared = Path(manifest["config"]).resolve()
    if (
        prepared != manifest_path.parent / "benchmark.py"
        or hashlib.sha256(prepared.read_bytes()).hexdigest() != manifest["config_sha256"]
    ):
        raise ValueError("Prepared configuration identity changed")
    work_dir = Path(manifest["work_dir"]).resolve()
    if work_dir != (manifest_path.parent / "results").resolve():
        raise ValueError("Manifest work directory is not private to this preparation")
    log = Path(args.log).read_text(encoding="utf-8", errors="replace")
    log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log)
    locations = {
        match.strip().removesuffix(".") for match in re.findall(r"Performance Result files located in ([^\r\n]+)", log)
    }
    if len(locations) != 1:
        raise ValueError("Current execution log must identify exactly one performance result directory")
    directory = Path(next(iter(locations))).resolve()
    if not directory.is_relative_to(work_dir):
        raise ValueError("Reported result directory is outside this execution's work directory")
    source_json = (directory / manifest["result_json_name"]).resolve()
    source_csv = (directory / manifest["result_csv_name"]).resolve()
    if any(not source.is_relative_to(directory) or not source.is_file() for source in (source_json, source_csv)):
        raise ValueError("Reported result files are missing or outside their directory")
    json_bytes, csv_bytes = source_json.read_bytes(), source_csv.read_bytes()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("result.json", json_bytes), ("result.csv", csv_bytes)):
        with (output / name).open("xb") as destination:
            destination.write(content)
    result = dict(
        expected,
        case=args.case,
        benchmark=args.benchmark,
        source_result_json=str(source_json),
        source_result_csv=str(source_csv),
        result_json=str(output / "result.json"),
        result_csv=str(output / "result.csv"),
        result_json_sha256=hashlib.sha256(json_bytes).hexdigest(),
        result_csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
    )
    argv = [
        sys.executable,
        "-m",
        "tools.nightly_cli",
        "verify",
        "--config",
        str(Path(args.config).resolve()),
        "--case",
        args.case,
        "--benchmark",
        args.benchmark,
        "--result-json",
        result["result_json"],
        "--result-csv",
        result["result_csv"],
        "--output-file",
        str(output / "verification.json"),
    ]
    script = (
        "#!/usr/bin/env bash\nset -euo pipefail\ncd -- "
        + shlex.quote(str(Path(__file__).resolve().parent.parent))
        + "\nexec "
        + shlex.join(argv)
        + "\n"
    )
    result["script"] = write_private(output, "verify.sh", script)
    write_private(output, "result-location.json", json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser("prepare", help="Write private AISBench configuration; never execute it")
    prepare_parser.add_argument("--benchmark-home", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--model-path", required=True)
    prepare_parser.add_argument("--dataset-path", required=True)
    prepare_parser.add_argument("--host", required=True)
    prepare_parser.add_argument("--port", required=True, type=int)
    server_parser = actions.add_parser("server-command", help="Write a vllm serve script; never execute it")
    server_parser.add_argument("--output-dir", required=True)
    server_parser.add_argument("--model-path", required=True)
    server_parser.add_argument("--host", required=True)
    server_parser.add_argument("--port", required=True, type=int)
    verify_parser = actions.add_parser("verify", help="Verify explicitly named existing JSON and CSV results")
    verify_parser.add_argument("--result-json", required=True)
    verify_parser.add_argument("--result-csv", required=True)
    verify_parser.add_argument("--output-file")
    locate_parser = actions.add_parser(
        "locate-results", help="Locate this run's results and write an explicit verify script"
    )
    locate_parser.add_argument("--manifest", required=True)
    locate_parser.add_argument("--log", required=True)
    locate_parser.add_argument("--output-dir", required=True)
    for action in (prepare_parser, server_parser, verify_parser, locate_parser):
        action.add_argument("--config", required=True)
        action.add_argument("--case", required=True)
        action.add_argument("--benchmark", default="perf")
    args = parser.parse_args(argv)
    try:
        case, config = select_case(args.config, args.case, args.benchmark)
        if hasattr(args, "port") and not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        if args.action == "prepare":
            result = prepare(args, case, config)
        elif args.action == "server-command":
            result = server_command(args, case, config)
        elif args.action == "locate-results":
            result = locate_results(args, config)
        else:
            result = verify(args, config)
        print(json.dumps(result))
        return 1 if result.get("verdict") == "failed" else 0
    except (
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OSError,
        SyntaxError,
        StopIteration,
        IndexError,
        yaml.YAMLError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
