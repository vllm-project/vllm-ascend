from __future__ import annotations

from io import StringIO
from pathlib import Path
from textwrap import dedent

import pytest

from tools.docs_codegen.cli import main
from tools.docs_codegen.errors import DocsCodegenError
from tools.docs_codegen.generator import GeneratorService
from tools.docs_codegen.scanner import BlockScanner


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")


# Smart UT only needs the top-level decorator shape; keep this pure tools test
# independent from tests.ut.conftest and vllm_ascend imports.
def npu_test(*, npu_type: str):
    if npu_type != "cpu":
        raise ValueError(f"unsupported docs_codegen UT target: {npu_type}")

    def decorator(func):
        return func

    return decorator


def _write_single_node_yaml(tmp_path: Path) -> Path:
    case_path = tmp_path / "cases" / "single_node.yaml"
    _write_text(
        case_path,
        """
        test_cases:
          - name: default-case
            model: default/model
            envs:
              SERVER_PORT: 8123
            server_cmd: []
          - name: selected-case
            model: "Qwen/Test Model"
            envs:
              HCCL_BUFFSIZE: 1024
              PROMPT: "hello world"
              SERVER_PORT: DEFAULT_PORT
            server_cmd:
              - "--tensor-parallel-size"
              - 2
              - "--kv-transfer-config"
              - '{"foo": "bar", "enabled": true}'
            server_cmd_extra: "--trust-remote-code --enable-expert-parallel"
        """,
    )
    return case_path.relative_to(tmp_path)


def _write_multi_node_yaml(tmp_path: Path, *, invalid_command: bool = False) -> Path:
    case_path = tmp_path / "cases" / "multi_node.yaml"
    second_command = (
        '"vllm serve multi-node/model extra-positional"'
        if invalid_command
        else """
              - vllm
              - serve
              - multi-node/model
              - "--headless"
              - "--port"
              - "$SERVER_PORT"
        """
    )
    _write_text(
        case_path,
        f"""
        deployment:
          - envs:
              LOCAL_IP: 127.0.0.1
            server_cmd: "vllm serve first-host --port 8000"
          - envs:
              MASTER_IP: 10.0.0.1
              SERVER_PORT: 9000
            server_cmd: {second_command.rstrip()}
        """,
    )
    return case_path.relative_to(tmp_path)


def _write_model_code_doc(tmp_path: Path, content: str, *, name: str = "Demo.md") -> Path:
    doc_path = tmp_path / "docs" / "models" / name
    _write_text(doc_path, content)
    return doc_path.relative_to(tmp_path)


def _generate_block(tmp_path: Path, doc_path: Path, block_name: str) -> str:
    service = GeneratorService(artifact_root="artifacts")
    return service.generate_block(doc_path, block_name, dry_run=True)[1].content


# Explicit CPU smart-UT routing keeps this guard out of the --run-all-cpu bucket.
@npu_test(npu_type="cpu")
def test_block_scanner_parses_metadata_and_trims_raw_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    single_yaml = _write_single_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        # Demo

        ```{{model-code}}
        :block_name: single
        :converter_tag: single_node
        :test_case_path: {single_yaml}
        :case_index: 1

        set -eux
        {{{{ generated }}}}

        ```
        """,
    )
    monkeypatch.chdir(tmp_path)

    blocks = BlockScanner().scan_document_blocks(doc_path)

    assert len(blocks) == 1
    block = blocks[0]
    assert block.doc_path == doc_path
    assert block.block_name == "single"
    assert block.converter_tag == "single_node"
    assert block.test_case_path == single_yaml.as_posix()
    assert block.extra_options == (("case_index", "1"),)
    assert block.directive_line == 3
    assert block.raw_block_lines == ("set -eux", "{{ generated }}")


@npu_test(npu_type="cpu")
def test_block_scanner_rejects_duplicate_block_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    single_yaml = _write_single_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: serve
        :converter_tag: single_node
        :test_case_path: {single_yaml}
        ```

        ```{{model-code}}
        :block_name: serve
        :converter_tag: single_node
        :test_case_path: {single_yaml}
        ```
        """,
        name="Duplicate.md",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DocsCodegenError) as exc_info:
        BlockScanner().scan_document_blocks(doc_path)

    error_message = str(exc_info.value)
    assert "docs/models/Duplicate.md:7: model-code generation error" in error_message
    assert "block_name: serve" in error_message
    assert "duplicated block_name 'serve'" in error_message
    assert "previous declaration is on line 1" in error_message


@npu_test(npu_type="cpu")
def test_block_scanner_rejects_unsupported_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    single_yaml = _write_single_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: serve
        :converter_tag: single_node
        :test_case_path: {single_yaml}
        :unknown_option: value
        ```
        """,
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DocsCodegenError) as exc_info:
        BlockScanner().scan_document_blocks(doc_path)

    assert "unsupported metadata: unknown_option" in str(exc_info.value)


@npu_test(npu_type="cpu")
def test_single_node_converter_uses_case_index_defaults_and_extra_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    single_yaml = _write_single_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: single
        :converter_tag: single_node
        :test_case_path: {single_yaml}
        :case_index: 1

        set -eux
        {{{{ generated }}}}
        ```
        """,
    )
    monkeypatch.chdir(tmp_path)

    script = _generate_block(tmp_path, doc_path, "single")

    assert script.startswith("set -eux\nexport HCCL_BUFFSIZE=1024")
    assert 'export PROMPT="hello world"' in script
    assert "export SERVER_PORT=8000" in script
    assert "ignored/model" not in script
    assert "vllm serve 'Qwen/Test Model' \\" in script
    assert "--tensor-parallel-size 2 \\" in script
    assert '"foo": "bar"' in script
    assert '"enabled": true' in script
    assert "--trust-remote-code \\" in script
    assert "--enable-expert-parallel" in script


@npu_test(npu_type="cpu")
def test_single_node_converter_defaults_to_first_case_and_preserves_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    single_yaml = _write_single_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: default_case
        :converter_tag: single_node
        :test_case_path: {single_yaml}
        ```
        """,
    )
    monkeypatch.chdir(tmp_path)

    script = _generate_block(tmp_path, doc_path, "default_case")

    assert script == "export SERVER_PORT=8123\n\nvllm serve default/model\n"


@npu_test(npu_type="cpu")
def test_single_node_converter_reports_invalid_case_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    single_yaml = _write_single_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: missing_case
        :converter_tag: single_node
        :test_case_path: {single_yaml}
        :case_index: 3
        ```
        """,
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DocsCodegenError) as exc_info:
        _generate_block(tmp_path, doc_path, "missing_case")

    assert "case_index 3 is out of range for 'test_cases' with 2 items" in str(exc_info.value)


@npu_test(npu_type="cpu")
def test_multi_node_converter_uses_host_index_and_token_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    multi_yaml = _write_multi_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: worker
        :converter_tag: multi_node
        :test_case_path: {multi_yaml}
        :host_index: 1
        ```
        """,
    )
    monkeypatch.chdir(tmp_path)

    script = _generate_block(tmp_path, doc_path, "worker")

    assert script.startswith("export MASTER_IP=10.0.0.1\nexport SERVER_PORT=9000")
    assert "vllm serve multi-node/model \\" in script
    assert "--headless \\" in script
    assert "--port $SERVER_PORT" in script
    assert "first-host" not in script


@npu_test(npu_type="cpu")
def test_multi_node_converter_requires_host_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    multi_yaml = _write_multi_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: worker
        :converter_tag: multi_node
        :test_case_path: {multi_yaml}
        ```
        """,
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DocsCodegenError) as exc_info:
        _generate_block(tmp_path, doc_path, "worker")

    assert "converter_tag 'multi_node' requires host_index" in str(exc_info.value)


@npu_test(npu_type="cpu")
def test_multi_node_converter_rejects_extra_positional_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    multi_yaml = _write_multi_node_yaml(tmp_path, invalid_command=True)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: worker
        :converter_tag: multi_node
        :test_case_path: {multi_yaml}
        :host_index: 1
        ```
        """,
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DocsCodegenError) as exc_info:
        _generate_block(tmp_path, doc_path, "worker")

    assert "unsupported positional argument 'extra-positional'" in str(exc_info.value)


@npu_test(npu_type="cpu")
def test_generator_service_writes_selected_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    single_yaml = _write_single_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: default_case
        :converter_tag: single_node
        :test_case_path: {single_yaml}
        ```
        """,
    )
    monkeypatch.chdir(tmp_path)

    service = GeneratorService(artifact_root="artifacts")
    output_path, generated_script = service.generate_block(doc_path, "default_case", dry_run=False)

    assert output_path == Path("artifacts/Demo/default_case.sh")
    assert output_path.read_text(encoding="utf-8") == generated_script.content
    assert generated_script.content == "export SERVER_PORT=8123\n\nvllm serve default/model\n"


@npu_test(npu_type="cpu")
def test_cli_generates_block_to_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    single_yaml = _write_single_node_yaml(tmp_path)
    doc_path = _write_model_code_doc(
        tmp_path,
        f"""
        ```{{model-code}}
        :block_name: default_case
        :converter_tag: single_node
        :test_case_path: {single_yaml}
        ```
        """,
    )
    monkeypatch.chdir(tmp_path)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(["--block", f"{doc_path}::default_case", "--dry-run", "--stdout"], stdout=stdout, stderr=stderr)

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert stdout.getvalue() == (
        "docs/_build/doc_codegen/Demo/default_case.sh\nexport SERVER_PORT=8123\n\nvllm serve default/model\n"
    )


@npu_test(npu_type="cpu")
def test_cli_rejects_invalid_block_reference():
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(["--block", "docs/model.md"], stdout=stdout, stderr=stderr)

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "block reference must use '<doc_path>::<block_name>'" in stderr.getvalue()
