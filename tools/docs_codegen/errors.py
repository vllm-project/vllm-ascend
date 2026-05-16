from __future__ import annotations

from pathlib import Path
from typing import Any


class DocsCodegenError(RuntimeError):
    """Raised when docs code generation fails."""

    def __init__(
        self,
        message: str,
        *,
        doc_path: Path | None = None,
        line: int | None = None,
        test_case_path: str | None = None,
        block_name: str | None = None,
        converter_tag: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.doc_path = doc_path
        self.line = line
        self.test_case_path = test_case_path
        self.block_name = block_name
        self.converter_tag = converter_tag

    def __str__(self) -> str:
        header = "model-code generation error"
        if self.doc_path is not None:
            header = self.doc_path.as_posix()
            if self.line is not None:
                header = f"{header}:{self.line}"
            header = f"{header}: model-code generation error"

        lines = [header]
        if self.block_name:
            lines.append(f"  block_name: {self.block_name}")
        if self.test_case_path:
            lines.append(f"  test_case_path: {self.test_case_path}")
        if self.converter_tag:
            lines.append(f"  converter_tag: {self.converter_tag}")
        lines.append(f"  detail: {self.message}")
        return "\n".join(lines)


def make_docs_codegen_error(
    message: str,
    *,
    block: Any | None = None,
    doc_path: Path | None = None,
    line: int | None = None,
    test_case_path: str | None = None,
    block_name: str | None = None,
    converter_tag: str | None = None,
) -> DocsCodegenError:
    if block is not None:
        doc_path = getattr(block, "doc_path", doc_path)
        line = getattr(block, "directive_line", line)
        test_case_path = getattr(block, "test_case_path", test_case_path)
        block_name = getattr(block, "block_name", block_name)
        converter_tag = getattr(block, "converter_tag", converter_tag)

    return DocsCodegenError(
        message,
        doc_path=doc_path,
        line=line,
        test_case_path=test_case_path,
        block_name=block_name,
        converter_tag=converter_tag,
    )
