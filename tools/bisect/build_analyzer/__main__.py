"""Inspect one or two commits' hardware-specific Docker build semantics."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from tools.bisect.build_analyzer import analyze_commit, analyze_directory, compare_builds, plan_transition
from tools.bisect.build_analyzer.base_image import from_reference
from tools.bisect.build_analyzer.diff import package_state_changes
from tools.bisect.build_analyzer.planner import plan_target_image_transition
from tools.bisect.build_analyzer.registry_image import RegistryImageResolver


def _write_json(path: Path, content: dict[str, object], run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{run_id}.tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--compare-commit")
    parser.add_argument("--first-dir", type=Path)
    parser.add_argument("--second-dir", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--soc", required=True, choices=("a2", "a3", "310p", "a5"))
    parser.add_argument("--arch", required=True, choices=("aarch64", "x86_64"))
    parser.add_argument("--os", default="ubuntu", dest="os_variant")
    parser.add_argument("--build-arg", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--inspect-base-images", action="store_true")
    parser.add_argument("--image-cache-dir", type=Path, default=Path.home() / ".cache" / "nightly_bisect" / "images")
    args = parser.parse_args()
    build_args = {}
    for assignment in args.build_arg:
        name, separator, value = assignment.partition("=")
        if not separator or not name:
            parser.error(f"invalid --build-arg {assignment!r}; expected NAME=VALUE")
        build_args[name] = value
    overrides = build_args or None
    if args.first_dir or args.second_dir:
        if not args.first_dir or not args.second_dir:
            parser.error("--first-dir and --second-dir must be supplied together")
        if args.compare_commit:
            parser.error("--compare-commit cannot be combined with directory comparison")
        before = analyze_directory(args.first_dir, args.soc, args.arch, args.os_variant, overrides)
        after = analyze_directory(args.second_dir, args.soc, args.arch, args.os_variant, overrides)
        result: dict[str, object] = {
            "first_directory": str(args.first_dir.resolve()),
            "second_directory": str(args.second_dir.resolve()),
            "analysis": asdict(before),
        }
    else:
        before = analyze_commit(args.repo, args.commit, args.soc, args.arch, args.os_variant, overrides)
        result = {"commit": args.commit, "analysis": asdict(before)}
        after = (
            analyze_commit(args.repo, args.compare_commit, args.soc, args.arch, args.os_variant, overrides)
            if args.compare_commit
            else None
        )
        if args.compare_commit:
            result["compare_commit"] = args.compare_commit
    if after is not None:
        run_id = uuid4().hex
        result["second_analysis"] = asdict(after)
        base_delta = None
        target_replay = None
        if args.inspect_base_images:
            before_base = next((from_reference(op.value) for op in before.operations if op.kind == "from"), None)
            after_base = next((from_reference(op.value) for op in after.operations if op.kind == "from"), None)
            if before_base is None or after_base is None:
                parser.error("both builds must have a FROM image to inspect")
            resolver = RegistryImageResolver(args.image_cache_dir)
            if before_base != after_base:
                reference, history = resolver.resolve_history(after_base, args.arch)
                target_replay = plan_target_image_transition(before, after, reference, history)
            result["base_image_delta"] = (
                {
                    "before": {"reference": base_delta.before.reference, "digest": base_delta.before.digest},
                    "after": {"reference": base_delta.after.reference, "digest": base_delta.after.digest},
                    "changed_files": list(base_delta.changed_files),
                    "changed_config": list(base_delta.changed_config),
                }
                if base_delta is not None
                else None
            )
        changes = compare_builds(before, after, base_delta)
        result["comparison"] = [
            {"change": kind, "before": asdict(old) if old else None, "after": asdict(new) if new else None}
            for kind, old, new in changes
        ]
        result["package_changes"] = [asdict(change) for change in package_state_changes(before, after)]
        transition = target_replay or plan_transition(before, after, base_delta)
        result["plan"] = asdict(transition)
        direction = (
            f"{args.first_dir.resolve()} -> {args.second_dir.resolve()}"
            if args.first_dir and args.second_dir
            else f"{args.commit} -> {args.compare_commit}"
        )
        plan_document: dict[str, object] = {
            "direction": direction,
            "soc": args.soc,
            "architecture": args.arch,
            **asdict(transition),
        }
        if args.plan_output:
            _write_json(args.plan_output, plan_document, run_id)
        result["before_diagnostics"] = list(before.diagnostics)
        result["after_diagnostics"] = list(after.diagnostics)
        result["comparison_diagnostics"] = [
            {"change": kind, "before": old.value if old else None, "after": new.value if new else None}
            for kind, old, new in changes
            if (old or new).kind == "unresolved"
        ]
        if args.output_dir:
            output_dir = args.output_dir
            common = {"analysis_id": run_id, "soc": args.soc, "architecture": args.arch}
            first_source = str(args.first_dir.resolve()) if args.first_dir else args.commit
            second_source = str(args.second_dir.resolve()) if args.second_dir else args.compare_commit
            artifacts = {
                "first_ir.json": {**common, "source": first_source, "ir": asdict(before)},
                "second_ir.json": {**common, "source": second_source, "ir": asdict(after)},
                "comparison.json": {
                    **common,
                    "from": first_source,
                    "to": second_source,
                    "changes": result["comparison"],
                    "package_changes": result["package_changes"],
                    "diagnostic_changes": result["comparison_diagnostics"],
                    "base_image_delta": result.get("base_image_delta"),
                },
                "execution_plan.json": {**common, **plan_document},
            }
            for filename, content in artifacts.items():
                _write_json(output_dir / filename, content, run_id)
            print(
                json.dumps({"analysis_id": run_id, "output_dir": str(output_dir.resolve()), "files": list(artifacts)})
            )
            return
    elif args.plan_output or args.output_dir:
        parser.error("output files require a comparison target")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
