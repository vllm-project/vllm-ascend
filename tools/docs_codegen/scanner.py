from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import regex as re

from tools.docs_codegen.errors import make_docs_codegen_error
from tools.docs_codegen.utils import trim_blank_edges

MODEL_CODE_DEFAULTS_PATH = Path("docs/source/tutorials/models")
MODEL_CODE_REQUIRED_OPTION_NAMES = ("block_name", "converter_tag", "test_case_path")
MODEL_CODE_OPTION_NAMES = (*MODEL_CODE_REQUIRED_OPTION_NAMES, "case_index", "host_index")
MODEL_CODE_OPEN_RE = re.compile(r"^\s*```{model-code}\s*$")
MODEL_CODE_CLOSE_RE = re.compile(r"^\s*```\s*$")
MODEL_CODE_OPTION_RE = re.compile(r"^\s*:([A-Za-z0-9_-]+):\s*(.*?)\s*$")
BLOCK_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ModelCodeBlock:
    """One ``model-code`` block discovered in a documentation page."""

    doc_path: Path
    block_name: str
    converter_tag: str
    test_case_path: str
    extra_options: tuple[tuple[str, str], ...] = ()
    directive_line: int | None = None
    raw_block_lines: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.doc_path.as_posix(), self.block_name)

    def get_option(self, name: str) -> str | None:
        for key, value in self.extra_options:
            if key == name:
                return value
        return None


class BlockScanner:
    """Scan markdown files for ``model-code`` directives."""

    def __init__(self, *, documents_root: str | Path = MODEL_CODE_DEFAULTS_PATH) -> None:
        self.documents_root = Path(documents_root)

    def scan_default_blocks(self) -> list[ModelCodeBlock]:
        if not self.documents_root.exists():
            raise make_docs_codegen_error(
                "tutorials models directory does not exist",
                doc_path=self.documents_root,
            )

        blocks: list[ModelCodeBlock] = []
        for doc_path in sorted(self.documents_root.rglob("*.md")):
            blocks.extend(self.scan_document_blocks(doc_path))
        return blocks

    def scan_document_blocks(self, doc_path: str | Path) -> list[ModelCodeBlock]:
        repo_relative_doc_path = self._normalize_document_path(doc_path)
        if not repo_relative_doc_path.exists():
            raise make_docs_codegen_error("document file does not exist", doc_path=repo_relative_doc_path)

        lines = repo_relative_doc_path.read_text(encoding="utf-8").splitlines()
        blocks: list[ModelCodeBlock] = []
        line_index = 0

        while line_index < len(lines):
            if not MODEL_CODE_OPEN_RE.match(lines[line_index]):
                line_index += 1
                continue

            directive_line = line_index + 1
            line_index += 1
            options: dict[str, str] = {}
            body_lines: list[str] = []
            in_body = False

            while line_index < len(lines):
                line = lines[line_index]
                if MODEL_CODE_CLOSE_RE.match(line):
                    break

                option_match = MODEL_CODE_OPTION_RE.match(line)
                if not in_body and option_match:
                    options[option_match.group(1)] = option_match.group(2).strip()
                else:
                    in_body = True
                    body_lines.append(line)
                line_index += 1

            if line_index >= len(lines) or not MODEL_CODE_CLOSE_RE.match(lines[line_index]):
                raise make_docs_codegen_error(
                    "unclosed model-code directive fence",
                    doc_path=repo_relative_doc_path,
                    line=directive_line,
                )

            blocks.append(
                self.build_block(
                    options,
                    doc_path=repo_relative_doc_path,
                    directive_line=directive_line,
                    body_lines=body_lines,
                )
            )
            line_index += 1

        self._validate_unique_block_names(blocks)
        return blocks

    def select_document_blocks(self, doc_path: str | Path, block_name: str | None = None) -> list[ModelCodeBlock]:
        blocks = self.scan_document_blocks(doc_path)
        if block_name is None:
            return blocks

        selected_blocks = [block for block in blocks if block.block_name == block_name]
        if not selected_blocks:
            raise make_docs_codegen_error(
                f"block_name '{block_name}' not found in document",
                doc_path=self._normalize_document_path(doc_path),
            )
        return selected_blocks

    def build_block(
        self,
        options: Mapping[str, str],
        *,
        doc_path: str | Path,
        directive_line: int | None = None,
        body_lines: Sequence[str] = (),
    ) -> ModelCodeBlock:
        repo_relative_doc_path = self._normalize_document_path(doc_path)
        missing = [name for name in MODEL_CODE_REQUIRED_OPTION_NAMES if name not in options]
        if missing:
            raise make_docs_codegen_error(
                f"model-code block missing required metadata: {', '.join(missing)}",
                doc_path=repo_relative_doc_path,
                line=directive_line,
                test_case_path=options.get("test_case_path"),
                block_name=options.get("block_name"),
                converter_tag=options.get("converter_tag"),
            )

        extra = sorted(set(options).difference(MODEL_CODE_OPTION_NAMES))
        if extra:
            raise make_docs_codegen_error(
                f"model-code block contains unsupported metadata: {', '.join(extra)}",
                doc_path=repo_relative_doc_path,
                line=directive_line,
                test_case_path=options.get("test_case_path"),
                block_name=options.get("block_name"),
                converter_tag=options.get("converter_tag"),
            )

        normalized_options = {name: options[name].strip() for name in MODEL_CODE_OPTION_NAMES if name in options}
        empty_values = [name for name, value in normalized_options.items() if not value]
        if empty_values:
            raise make_docs_codegen_error(
                f"model-code block contains empty metadata: {', '.join(empty_values)}",
                doc_path=repo_relative_doc_path,
                line=directive_line,
                test_case_path=options.get("test_case_path"),
                block_name=options.get("block_name"),
                converter_tag=options.get("converter_tag"),
            )

        if not BLOCK_NAME_RE.fullmatch(normalized_options["block_name"]):
            raise make_docs_codegen_error(
                "block_name may only contain letters, numbers, dots, underscores, and dashes",
                doc_path=repo_relative_doc_path,
                line=directive_line,
                test_case_path=options.get("test_case_path"),
                block_name=options.get("block_name"),
                converter_tag=options.get("converter_tag"),
            )

        return ModelCodeBlock(
            doc_path=repo_relative_doc_path,
            block_name=normalized_options["block_name"],
            converter_tag=normalized_options["converter_tag"],
            test_case_path=normalized_options["test_case_path"],
            extra_options=tuple(
                (name, normalized_options[name])
                for name in MODEL_CODE_OPTION_NAMES
                if name not in MODEL_CODE_REQUIRED_OPTION_NAMES and name in normalized_options
            ),
            directive_line=directive_line,
            raw_block_lines=tuple(trim_blank_edges(body_lines)),
        )

    @staticmethod
    def _normalize_document_path(doc_path: str | Path) -> Path:
        candidate = Path(doc_path)
        if candidate.is_absolute():
            raise make_docs_codegen_error("document path must be repository-relative", doc_path=candidate)
        return candidate

    @staticmethod
    def _validate_unique_block_names(blocks: Sequence[ModelCodeBlock]) -> None:
        seen: dict[tuple[str, str], ModelCodeBlock] = {}
        for block in blocks:
            previous = seen.get(block.key)
            if previous is None:
                seen[block.key] = block
                continue

            raise make_docs_codegen_error(
                "duplicated block_name "
                f"'{block.block_name}' in document; previous declaration is on line {previous.directive_line}",
                block=block,
            )
