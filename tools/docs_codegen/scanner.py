from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import regex as re

from tools.docs_codegen.errors import DocsCodegenError, make_docs_codegen_error
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
        """Identity used to detect duplicate blocks: ``(doc_path, block_name)``."""
        return (self.doc_path.as_posix(), self.block_name)

    def get_option(self, name: str) -> str | None:
        """Return the value of an extra (non-required) directive option, or ``None``."""
        for key, value in self.extra_options:
            if key == name:
                return value
        return None


class BlockScanner:
    """Scan markdown files for ``model-code`` directives."""

    def __init__(
        self,
        *,
        documents_root: str | Path = MODEL_CODE_DEFAULTS_PATH,
        repo_root: str | Path | None = None,
    ) -> None:
        self.documents_root = Path(documents_root)
        self.repo_root = Path(repo_root) if repo_root is not None else None

    # -- Public API ----------------------------------------------------------

    def scan_default_blocks(self) -> list[ModelCodeBlock]:
        """Scan every markdown file under ``documents_root`` for model-code blocks."""
        base = self._base
        models_dir = base / self.documents_root
        if not models_dir.exists():
            raise make_docs_codegen_error(
                "tutorials models directory does not exist",
                doc_path=self.documents_root,
            )

        blocks: list[ModelCodeBlock] = []
        for absolute_doc_path in sorted(models_dir.rglob("*.md")):
            blocks.extend(self.scan_document_blocks(absolute_doc_path.relative_to(base)))
        return blocks

    def scan_document_blocks(self, doc_path: str | Path) -> list[ModelCodeBlock]:
        """Parse all model-code directive fences in a single markdown document."""
        repo_relative_doc_path = self._normalize_document_path(doc_path)
        absolute_doc_path = self._base / repo_relative_doc_path
        if not absolute_doc_path.exists():
            raise make_docs_codegen_error("document file does not exist", doc_path=repo_relative_doc_path)

        lines = absolute_doc_path.read_text(encoding="utf-8").splitlines()
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
        """Scan a document and optionally keep only the block named ``block_name``."""
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
        """Validate directive options and assemble a ``ModelCodeBlock``."""
        repo_relative_doc_path = self._normalize_document_path(doc_path)

        def fail(message: str) -> DocsCodegenError:
            return make_docs_codegen_error(
                message,
                doc_path=repo_relative_doc_path,
                line=directive_line,
                test_case_path=options.get("test_case_path"),
                block_name=options.get("block_name"),
                converter_tag=options.get("converter_tag"),
            )

        missing = [name for name in MODEL_CODE_REQUIRED_OPTION_NAMES if name not in options]
        if missing:
            raise fail(f"model-code block missing required metadata: {', '.join(missing)}")

        extra = sorted(set(options).difference(MODEL_CODE_OPTION_NAMES))
        if extra:
            raise fail(f"model-code block contains unsupported metadata: {', '.join(extra)}")

        normalized_options = {name: options[name].strip() for name in MODEL_CODE_OPTION_NAMES if name in options}
        empty_values = [name for name, value in normalized_options.items() if not value]
        if empty_values:
            raise fail(f"model-code block contains empty metadata: {', '.join(empty_values)}")

        if not BLOCK_NAME_RE.fullmatch(normalized_options["block_name"]):
            raise fail("block_name may only contain letters, numbers, dots, underscores, and dashes")

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

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _normalize_document_path(doc_path: str | Path) -> Path:
        """Validate that a document path is repository-relative and contained."""
        candidate = Path(doc_path)
        if candidate.is_absolute():
            raise make_docs_codegen_error("document path must be repository-relative", doc_path=candidate)
        if ".." in candidate.parts:
            raise make_docs_codegen_error("document path must stay within the repository", doc_path=candidate)
        return candidate

    @staticmethod
    def _validate_unique_block_names(blocks: Sequence[ModelCodeBlock]) -> None:
        """Reject documents that declare the same block_name twice."""
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

    # -- Path helpers --------------------------------------------------------

    @property
    def _base(self) -> Path:
        """Directory that repo-relative paths resolve against for filesystem I/O."""
        return self.repo_root if self.repo_root is not None else Path.cwd()
