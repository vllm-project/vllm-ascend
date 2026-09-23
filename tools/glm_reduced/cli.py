# SPDX-License-Identifier: Apache-2.0
"""Command line interface for the GLM reduced-checkpoint tool.

Subcommands:
  inventory      print the GLM model -> recipe inventory (markdown table)
  profiles       list reduction profiles and their rules
  plan           dry-run a reduction: tensor classification and config diff
  build          execute a reduction, writing the checkpoint + manifest
  verify         re-check a reduced checkpoint against its manifest

All subcommands are CPU-only and require no vllm/torch installation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import ReductionError
from .inventory import list_inventory, render_markdown, validate_inventory
from .profiles import get_profile, list_profiles, match_profile
from .reducer import (
    DEFAULT_MAX_SHARD_BYTES,
    execute_plan,
    load_source,
    load_source_selective,
    plan_reduction,
    required_shards,
    verify_reduced,
)


def _cmd_inventory(args: argparse.Namespace) -> int:
    problems = validate_inventory()
    if problems:
        for problem in problems:
            print(f"inventory inconsistency: {problem}", file=sys.stderr)
        return 1
    if args.format == "markdown":
        print(render_markdown())
    else:
        print(json.dumps([vars(entry) for entry in list_inventory()], indent=2, default=list))
    return 0


def _cmd_profiles(args: argparse.Namespace) -> int:
    for profile in list_profiles():
        workload = profile.precision_workload
        print(
            f"{profile.name}: family={profile.family} layout={profile.config_layout} "
            f"architectures={list(profile.architectures)} layer_count_keys={list(profile.layer_count_keys)} "
            f"min/default keep={profile.min_keep_layers}/{profile.default_keep_layers} "
            f"official_layers={profile.official_layers or 'varies'} "
            f"keep_mtp={profile.keep_mtp} include_vision={profile.include_vision} "
            f"precision_prompts={list(workload.prompt_lengths)}"
        )
    return 0


def _cmd_sources(args: argparse.Namespace) -> int:
    sources_path = Path(__file__).resolve().parent / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    if args.model:
        sources = [s for s in sources if s["model_id"] == args.model]
        if not sources:
            print(f"error: no pinned source descriptor for {args.model!r}", file=sys.stderr)
            return 2
    print(json.dumps(sources, indent=2))
    return 0


def _cmd_required_shards(args: argparse.Namespace) -> int:
    profile = get_profile(args.profile)
    result = required_shards(args.config, args.index, profile, args.layers)
    print(json.dumps(result, indent=2))
    return 0


def _load_and_plan(args: argparse.Namespace):
    if args.selective:
        config_path = Path(args.source) / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        profile = get_profile(args.profile) if args.profile else match_profile(config.get("architectures"))
        if profile is None:
            raise ReductionError(
                f"no profile matches architectures {config.get('architectures')}; "
                "pass --profile explicitly or extend tools/glm_reduced/profiles.py"
            )
        source = load_source_selective(args.source, profile, args.layers or profile.default_keep_layers)
    else:
        source = load_source(args.source)
        profile = get_profile(args.profile) if args.profile else match_profile(source.config.get("architectures"))
    if profile is None:
        raise ReductionError(
            f"no profile matches architectures {source.config.get('architectures')}; "
            "pass --profile explicitly or extend tools/glm_reduced/profiles.py"
        )
    plan = plan_reduction(
        source,
        profile,
        keep_layers=args.layers,
        allow_extra_tensors=args.allow_extra_tensors,
        truncate_unknown_arrays=args.truncate_unknown_arrays,
    )
    return source, plan


def _cmd_plan(args: argparse.Namespace) -> int:
    source, plan = _load_and_plan(args)
    kept_bytes = 0
    by_action: dict[str, int] = {}
    for action in plan.actions:
        by_action[action.action] = by_action.get(action.action, 0) + 1
        if action.action in ("keep", "remap"):
            shard = source.shards[source.weight_map[action.src_name]]
            kept_bytes += shard.tensors[action.src_name].nbytes
    summary = {
        "profile": plan.profile.name,
        "source_layers": plan.source_layers,
        "keep_layers": plan.keep_layers,
        "mtp_layers": plan.mtp_layers,
        "actions": by_action,
        "kept_bytes": kept_bytes,
        "warnings": plan.warnings,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    source, plan = _load_and_plan(args)
    manifest = execute_plan(plan, source, args.output, max_shard_bytes=args.max_shard_bytes)
    report = verify_reduced(args.output)
    print(json.dumps({"manifest": {"output": manifest["output"]["files"]}, "verify": report}, indent=2))
    if not report["ok"]:
        print("build finished but post-build verification FAILED", file=sys.stderr)
        return 1
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    report = verify_reduced(args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glm_reduced", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inventory", help="print the GLM model inventory")
    p.add_argument("--format", choices=("markdown", "json"), default="markdown")
    p.set_defaults(func=_cmd_inventory)

    p = sub.add_parser("profiles", help="list reduction profiles")
    p.set_defaults(func=_cmd_profiles)

    p = sub.add_parser("sources", help="print pinned public source descriptors (model IDs, revisions, URLs)")
    p.add_argument("--model", help="filter to one model id")
    p.set_defaults(func=_cmd_sources)

    p = sub.add_parser(
        "required-shards",
        help="list the shard files a reduction needs, from a downloaded config.json + index "
        "(selective-download support; avoids pulling multi-TB checkpoints blindly)",
    )
    p.add_argument("--config", required=True, help="path to the checkpoint's config.json")
    p.add_argument("--index", required=True, help="path to the checkpoint's safetensors index json")
    p.add_argument("--profile", required=True)
    p.add_argument("--layers", type=int, required=True)
    p.set_defaults(func=_cmd_required_shards)

    def add_build_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("source", help="source checkpoint directory (never modified)")
        p.add_argument("--profile", help="reduction profile name (default: auto-match by architectures)")
        p.add_argument("--layers", type=int, default=None, help="kept layer prefix (default: profile default)")
        p.add_argument(
            "--allow-extra-tensors",
            action="store_true",
            help="copy unrecognized tensors verbatim instead of failing (recorded in the manifest)",
        )
        p.add_argument(
            "--truncate-unknown-arrays",
            action="store_true",
            help="truncate unrecognized config lists of length num_hidden_layers instead of failing",
        )
        p.add_argument(
            "--selective",
            action="store_true",
            help="selective-download source: only shards needed for the reduction are present "
            "(dropped-only shards may be absent); full-source strictness applies otherwise",
        )

    p = sub.add_parser("plan", help="dry-run a reduction")
    add_build_args(p)
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("build", help="execute a reduction")
    add_build_args(p)
    p.add_argument("output", help="output directory (must not exist; published atomically)")
    p.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("verify", help="verify a reduced checkpoint against its manifest")
    p.add_argument("output")
    p.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ReductionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
