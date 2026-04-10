from __future__ import annotations

from pathlib import Path

from tools.docs_codegen.converters import BaseConverter, GeneratedScript, build_default_converters, get_converter
from tools.docs_codegen.errors import make_docs_codegen_error
from tools.docs_codegen.scanner import BlockScanner, ModelCodeBlock
from tools.docs_codegen.yaml_loader import YamlLoader

DEFAULT_ARTIFACT_ROOT = Path("docs/_build/doc_codegen")
GENERATED_SCRIPT_MARKER = "{{ generated }}"


class GeneratorService:
    """Shared generation pipeline used by both CLI and Sphinx."""

    def __init__(
        self,
        *,
        block_scanner: BlockScanner | None = None,
        yaml_loader: YamlLoader | None = None,
        converters: dict[str, BaseConverter] | None = None,
        artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ) -> None:
        self.block_scanner = block_scanner or BlockScanner()
        self.yaml_loader = yaml_loader or YamlLoader()
        self.converters = converters or build_default_converters()
        self.artifact_root = Path(artifact_root)

    def generate_all(self, *, dry_run: bool = False) -> list[tuple[Path, GeneratedScript]]:
        return self._generate_blocks(self.block_scanner.scan_default_blocks(), dry_run=dry_run)

    def generate_document(self, doc_path: str | Path, *, dry_run: bool = False) -> list[tuple[Path, GeneratedScript]]:
        return self._generate_blocks(self.block_scanner.scan_document_blocks(doc_path), dry_run=dry_run)

    def generate_block(
        self,
        doc_path: str | Path,
        block_name: str,
        *,
        dry_run: bool = False,
    ) -> tuple[Path, GeneratedScript]:
        generated_artifacts = self._generate_blocks(
            self.block_scanner.select_document_blocks(doc_path, block_name),
            dry_run=dry_run,
        )
        return generated_artifacts[0]

    def get_block(self, doc_path: str | Path, block_name: str) -> ModelCodeBlock:
        return self.block_scanner.select_document_blocks(doc_path, block_name)[0]

    def output_path_for(self, block: ModelCodeBlock) -> Path:
        return self.artifact_root / block.doc_path.stem / f"{block.block_name}.sh"

    def output_path_for_block(self, doc_path: str | Path, block_name: str) -> Path:
        return self.output_path_for(self.get_block(doc_path, block_name))

    def read_generated_script(self, block: ModelCodeBlock) -> GeneratedScript:
        output_path = self.output_path_for(block)
        if not output_path.exists():
            raise make_docs_codegen_error(
                f"generated artifact not found: {output_path}",
                block=block,
            )
        return GeneratedScript(content=output_path.read_text(encoding="utf-8"))

    def _generate_blocks(self, blocks: list[ModelCodeBlock], *, dry_run: bool) -> list[tuple[Path, GeneratedScript]]:
        generated_artifacts: list[tuple[Path, GeneratedScript]] = []
        for block in blocks:
            converter = get_converter(self.converters, block.converter_tag, block=block)
            loaded_yaml = self.yaml_loader.load(
                test_case_path=block.test_case_path,
                block=block,
            )
            generated_script = converter.convert(loaded_yaml, block=block)
            generated_script = self._merge_raw_block_with_generated_script(generated_script, block=block)
            self._validate_generated_script(generated_script, block=block)
            output_path = self.output_path_for(block)

            if not dry_run:
                self._write_script(output_path, generated_script)
            generated_artifacts.append((output_path, generated_script))
        return generated_artifacts

    @staticmethod
    def _merge_raw_block_with_generated_script(
        generated_script: GeneratedScript,
        *,
        block: ModelCodeBlock,
    ) -> GeneratedScript:
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
        if not generated_script.content.strip():
            raise make_docs_codegen_error("generated script content is empty", block=block)

    @staticmethod
    def _write_script(output_path: Path, generated_script: GeneratedScript) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(generated_script.content, encoding="utf-8")


def create_default_generator_service() -> GeneratorService:
    return GeneratorService()
