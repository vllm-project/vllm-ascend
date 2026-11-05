from __future__ import annotations

from pathlib import Path

from tools.docs_codegen.converters import BaseConverter, GeneratedScript, build_default_converters, get_converter
from tools.docs_codegen.errors import make_docs_codegen_error
from tools.docs_codegen.scanner import BlockScanner, ModelCodeBlock
from tools.docs_codegen.yaml_loader import YamlLoader

DEFAULT_ARTIFACT_ROOT = Path("docs/_build/doc_codegen")
GENERATED_SCRIPT_MARKER = "{{ generated }}"

# One generated artifact: its repo-relative output path and the rendered script.
GeneratedArtifact = tuple[Path, GeneratedScript]


class GeneratorService:
    """Shared generation pipeline used by both CLI and Sphinx."""

    def __init__(
        self,
        *,
        block_scanner: BlockScanner | None = None,
        yaml_loader: YamlLoader | None = None,
        converters: dict[str, BaseConverter] | None = None,
        artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
        repo_root: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else None
        self.block_scanner = block_scanner or BlockScanner(repo_root=self.repo_root)
        self.yaml_loader = yaml_loader or YamlLoader(repo_root=self.repo_root)
        self.converters = converters or build_default_converters()
        self.artifact_root = Path(artifact_root)

    # -- Public API ----------------------------------------------------------

    def generate_all(self, *, dry_run: bool = False) -> list[GeneratedArtifact]:
        """Generate artifacts for every model-code block under the documents root."""
        return self._generate_blocks(self.block_scanner.scan_default_blocks(), dry_run=dry_run)

    def generate_document(self, doc_path: str | Path, *, dry_run: bool = False) -> list[GeneratedArtifact]:
        """Generate artifacts for every model-code block in one document."""
        return self._generate_blocks(self.block_scanner.scan_document_blocks(doc_path), dry_run=dry_run)

    def generate_block(
        self,
        doc_path: str | Path,
        block_name: str,
        *,
        dry_run: bool = False,
    ) -> GeneratedArtifact:
        """Generate the artifact for a single named block in a document."""
        generated_artifacts = self._generate_blocks(
            self.block_scanner.select_document_blocks(doc_path, block_name),
            dry_run=dry_run,
        )
        return generated_artifacts[0]

    def read_generated_script(self, block: ModelCodeBlock) -> GeneratedScript:
        """Read a previously generated artifact from disk (used by the Sphinx directive)."""
        output_path = self.output_path_for(block)
        absolute_output_path = self._absolute_output_path(output_path)
        if not absolute_output_path.exists():
            raise make_docs_codegen_error(
                f"generated artifact not found: {output_path}",
                block=block,
            )
        return GeneratedScript(content=absolute_output_path.read_text(encoding="utf-8"))

    def output_path_for(self, block: ModelCodeBlock) -> Path:
        """Repo-relative artifact path: ``<artifact_root>/<doc_stem>/<block_name>.sh``."""
        return self.artifact_root / block.doc_path.stem / f"{block.block_name}.sh"

    # -- Generation pipeline -------------------------------------------------

    def _generate_blocks(self, blocks: list[ModelCodeBlock], *, dry_run: bool) -> list[GeneratedArtifact]:
        """Convert each block to a script, merge any raw body, and (unless dry-run) write it."""
        generated_artifacts: list[GeneratedArtifact] = []
        for block in blocks:
            converter = get_converter(self.converters, block.converter_tag, block=block)
            loaded_yaml = self.yaml_loader.load(
                test_case_path=block.test_case_path,
                block=block,
            )
            generated_script = converter.convert(loaded_yaml, block=block)
            generated_script = self._apply_block_body(generated_script, block=block)
            self._validate_generated_script(generated_script, block=block)
            output_path = self.output_path_for(block)

            if not dry_run:
                self._write_script(self._absolute_output_path(output_path), generated_script)
            generated_artifacts.append((output_path, generated_script))
        return generated_artifacts

    @staticmethod
    def _apply_block_body(
        generated_script: GeneratedScript,
        *,
        block: ModelCodeBlock,
    ) -> GeneratedScript:
        """Splice the converter output into the block body (at ``{{ generated }}`` or appended)."""
        if not block.raw_block_lines:
            return generated_script

        raw_block_content = "\n".join(block.raw_block_lines)
        generated_content = generated_script.content.rstrip()
        if GENERATED_SCRIPT_MARKER in raw_block_content:
            content = raw_block_content.replace(GENERATED_SCRIPT_MARKER, generated_content)
        else:
            content = f"{raw_block_content.rstrip()}\n\n{generated_content}"

        return GeneratedScript(content=content.rstrip() + "\n", language=generated_script.language)

    @staticmethod
    def _validate_generated_script(generated_script: GeneratedScript, *, block: ModelCodeBlock) -> None:
        """Guard against a converter producing an empty artifact."""
        if not generated_script.content.strip():
            raise make_docs_codegen_error("generated script content is empty", block=block)

    @staticmethod
    def _write_script(output_path: Path, generated_script: GeneratedScript) -> None:
        """Write an artifact to ``output_path``, creating parent directories."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(generated_script.content, encoding="utf-8")

    # -- Path helpers --------------------------------------------------------

    @property
    def _base(self) -> Path:
        """Directory that the repo-relative artifact path resolves against for I/O."""
        return self.repo_root if self.repo_root is not None else Path.cwd()

    def _absolute_output_path(self, output_path: Path) -> Path:
        """Anchor a repo-relative output path to ``_base`` for filesystem reads/writes."""
        return output_path if output_path.is_absolute() else self._base / output_path


def create_default_generator_service(repo_root: str | Path | None = None) -> GeneratorService:
    """Build a ``GeneratorService`` with default scanner/loader/converters."""
    return GeneratorService(repo_root=repo_root)
