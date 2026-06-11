from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.errors import SphinxError
from sphinx.util.docutils import SphinxDirective

from tools.docs_codegen.errors import DocsCodegenError
from tools.docs_codegen.generator import GeneratorService, create_default_generator_service
from tools.docs_codegen.scanner import BlockScanner, ModelCodeBlock

# Anchor all repo-relative paths here instead of relying on the process CWD: the
# docs are built from ``docs/`` (see docs/Makefile, SOURCEDIR=source), so the
# generator must resolve paths against the repo root regardless of where
# sphinx-build was launched.
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_block_from_options(
    *,
    doc_path: Path,
    options: Mapping[str, str],
    directive_line: int | None = None,
    body_lines: list[str] | None = None,
    block_scanner: BlockScanner | None = None,
) -> ModelCodeBlock:
    """Build a ``ModelCodeBlock`` from directive options (no filesystem scan)."""
    scanner = block_scanner or BlockScanner(repo_root=REPO_ROOT)
    return scanner.build_block(options, doc_path=doc_path, directive_line=directive_line, body_lines=body_lines or ())


def render_generated_script(
    block: ModelCodeBlock,
    *,
    service: GeneratorService | None = None,
) -> nodes.literal_block:
    """Read the pre-generated artifact for a block and wrap it in a docutils literal block."""
    generator_service = service or create_default_generator_service(repo_root=REPO_ROOT)
    script = generator_service.read_generated_script(block)
    literal = nodes.literal_block(script.content, script.content)
    literal["language"] = script.language
    return literal


class ModelCodeDirective(SphinxDirective):
    """Import a pre-generated shell script and render it as a code block."""

    has_content = True
    option_spec = {
        "block_name": directives.unchanged_required,
        "converter_tag": directives.unchanged_required,
        "test_case_path": directives.unchanged_required,
        "case_index": directives.unchanged,
        "host_index": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        """Resolve the current document's block and emit its rendered code block."""
        source_relative_doc_path = Path(self.env.doc2path(self.env.docname, base=False))
        doc_path = Path("docs/source") / source_relative_doc_path

        try:
            block = build_block_from_options(
                doc_path=doc_path,
                options=self.options,
                directive_line=self.lineno,
                body_lines=list(self.content),
            )
            return [render_generated_script(block)]
        except DocsCodegenError as exc:
            raise self.error(str(exc)) from exc


def on_builder_inited(app) -> None:
    """Sphinx ``builder-inited`` hook: regenerate all artifacts before the build reads them."""
    del app
    try:
        create_default_generator_service(repo_root=REPO_ROOT).generate_all()
    except DocsCodegenError as exc:
        raise SphinxError(str(exc)) from exc


def setup(app):
    """Sphinx extension entry point: register the directive and the build-init hook."""
    app.add_directive("model-code", ModelCodeDirective)
    app.connect("builder-inited", on_builder_inited)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
