import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[skip] {path.name}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(data, dict) or "meta" not in data:
        print(f"[skip] {path.name}: invalid dump format")
        return None
    return data


def _tensor_to_int(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.item())
        return None
    if isinstance(value, int):
        return value
    return None


def cmd_scan(args: argparse.Namespace) -> None:
    dump_dir = Path(args.dump_dir)
    paths = sorted(dump_dir.glob("identical_req_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No identical-request dumps found under {dump_dir}")

    matches: list[tuple[Path, dict[str, Any]]] = []
    grouped: dict[tuple[str, int | None, int | None, str], list[tuple[Path, int, int]]] = defaultdict(list)

    for path in paths:
        data = _load(path)
        if data is None:
            continue
        meta = data.get("meta", {})
        if meta.get("kind") != "tensor":
            continue
        stage = meta.get("stage")
        if args.stage and stage != args.stage:
            continue
        if args.name_contains and args.name_contains not in str(meta.get("name", "")):
            continue

        tensors = data.get("tensors", {})
        num_decodes = _tensor_to_int(tensors.get("num_decodes"))
        num_prefills = _tensor_to_int(tensors.get("num_prefills"))
        if num_decodes is None:
            continue
        if num_decodes < args.min_decodes:
            continue
        if args.require_pure_decode and num_prefills != 0:
            continue

        matches.append((path, data))
        key = (stage, meta.get("decode_step"), meta.get("tp_rank"), str(meta.get("name")))
        grouped[key].append((path, num_decodes, -1 if num_prefills is None else num_prefills))

    print(f"matched_files={len(matches)}")
    print("=" * 140)
    print("GROUP SUMMARY")
    for key in sorted(grouped):
        items = grouped[key]
        stage, step, tp_rank, name = key
        num_decodes_values = sorted({item[1] for item in items})
        num_prefills_values = sorted({item[2] for item in items})
        print(
            f"stage={stage} step={step} tp={tp_rank} name={name} "
            f"count={len(items)} num_decodes={num_decodes_values} num_prefills={num_prefills_values}"
        )

    if not matches:
        return

    print("=" * 140)
    print("MATCHED FILES")
    for path, data in matches:
        meta = data["meta"]
        tensors = data.get("tensors", {})
        print(path.name)
        print(
            f"  stage={meta.get('stage')} step={meta.get('decode_step')} dp={meta.get('dp_rank')} tp={meta.get('tp_rank')}"
        )
        print(f"  name={meta.get('name')}")
        print(f"  num_decodes={_tensor_to_int(tensors.get('num_decodes'))}")
        print(f"  num_prefills={_tensor_to_int(tensors.get('num_prefills'))}")
        print(f"  decode_req_ids={tensors.get('decode_req_ids')}")
        print(f"  prefill_req_ids={tensors.get('prefill_req_ids')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find identical-request dumps with multiple decode requests.")
    parser.add_argument("dump_dir", help="Directory containing identical-request .pt dumps.")
    parser.add_argument(
        "--stage",
        choices=["prefill", "decode", "mixed"],
        default="decode",
        help="Stage filter. Default: decode",
    )
    parser.add_argument(
        "--min-decodes",
        type=int,
        default=2,
        help="Minimum num_decodes to report. Default: 2",
    )
    parser.add_argument(
        "--require-pure-decode",
        action="store_true",
        help="Only report files with num_prefills == 0.",
    )
    parser.add_argument(
        "--name-contains",
        help="Optional substring filter on meta.name, e.g. layers.0.self_attn.attn",
    )
    parser.set_defaults(func=cmd_scan)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
