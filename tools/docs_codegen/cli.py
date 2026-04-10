from __future__ import annotations

import argparse
import sys
from typing import TextIO

if __name__ == "__main__":
    # Keep `python3 tools/docs_codegen/cli.py ...` working from the repository root.
    sys.path.insert(0, ".")

from tools.docs_codegen.errors import DocsCodegenError


def build_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description="Generate shell code blocks from model-code directives.")
    selection_group = arg_parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--doc",
        dest="doc_path",
        help="Generate all blocks from one repository-relative markdown path.",
    )
    selection_group.add_argument(
        "--block",
        dest="block_ref",
        help="Block reference in '<doc_path>::<block_name>' form.",
    )
    _add_generate_flags(arg_parser)

    return arg_parser


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = build_arg_parser().parse_args(argv)

    try:
        return _handle_generate(args, stdout=stdout)
    except DocsCodegenError as exc:
        print(exc, file=stderr)
        return 1


def _add_generate_flags(arg_parser: argparse.ArgumentParser) -> None:
    arg_parser.add_argument("--stdout", action="store_true", help="Print generated content after the output path.")
    arg_parser.add_argument("--dry-run", action="store_true", help="Generate content without writing files.")


def _handle_generate(args: argparse.Namespace, *, stdout: TextIO) -> int:
    from tools.docs_codegen.generator import create_default_generator_service

    service = create_default_generator_service()
    if args.doc_path is not None:
        generated_artifacts = service.generate_document(args.doc_path, dry_run=args.dry_run)
    elif args.block_ref is not None:
        doc_path, block_name = _parse_block_ref(args.block_ref)
        generated_artifacts = [service.generate_block(doc_path, block_name, dry_run=args.dry_run)]
    else:
        generated_artifacts = service.generate_all(dry_run=args.dry_run)

    for output_path, script in generated_artifacts:
        print(output_path, file=stdout)
        if args.stdout:
            print(script.content.rstrip(), file=stdout)

    return 0


def _parse_block_ref(block_ref: str) -> tuple[str, str]:
    if "::" not in block_ref:
        raise DocsCodegenError("block reference must use '<doc_path>::<block_name>'")

    doc_path, block_name = block_ref.rsplit("::", 1)
    if not doc_path or not block_name:
        raise DocsCodegenError("block reference must use '<doc_path>::<block_name>'")
    return doc_path, block_name


if __name__ == "__main__":
    raise SystemExit(main())
