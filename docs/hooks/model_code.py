"""MkDocs hook: process model-code blocks before build.

This hook runs the docs_codegen generator to produce shell scripts,
then replaces model-code directive fences in markdown with the
generated script content.
"""

import sys
from pathlib import Path

import regex as re

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.docs_codegen.generator import create_default_generator_service  # noqa: E402
from tools.docs_codegen.scanner import BlockScanner  # noqa: E402

MODEL_CODE_FENCE_RE = re.compile(r"```{model-code}\s*\n(.*?)```", re.DOTALL)


def _process_model_code_blocks(markdown: str, doc_path: str) -> str:
    """Replace model-code fences with generated script content."""
    scanner = BlockScanner(repo_root=REPO_ROOT)
    blocks = scanner.scan_document_blocks(doc_path)

    if not blocks:
        return markdown

    service = create_default_generator_service(repo_root=REPO_ROOT)

    def replace_block(match):
        body = match.group(1)
        # Parse options from body
        options = {}
        for line in body.strip().splitlines():
            opt_match = re.match(r":(\w[-\w]*):\s*(.*)", line)
            if opt_match:
                options[opt_match.group(1)] = opt_match.group(2).strip()

        block_name = options.get("block_name")
        if not block_name:
            return match.group(0)

        # Find matching block
        for block in blocks:
            if block.block_name == block_name:
                try:
                    script = service.read_generated_script(block)
                    return f"```bash\n{script.content}\n```"
                except Exception:
                    return match.group(0)

        return match.group(0)

    return MODEL_CODE_FENCE_RE.sub(replace_block, markdown)


def on_page_markdown(markdown, page, config, files):
    """MkDocs hook: replace model-code blocks with generated scripts."""
    doc_path = str(Path(page.file.abs_src_path).relative_to(REPO_ROOT))
    return _process_model_code_blocks(markdown, doc_path)


def on_pre_build(config):
    """MkDocs hook: generate all model-code artifacts before build."""
    from tools.docs_codegen.errors import DocsCodegenError

    try:
        service = create_default_generator_service(repo_root=REPO_ROOT)
        service.generate_all()
        print("[model-code] Artifacts generated")
    except DocsCodegenError as exc:
        print(f"[model-code] Warning: {exc}")
