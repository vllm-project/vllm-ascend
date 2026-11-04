# SPDX-License-Identifier: Apache-2.0
"""Cold selection of the paired A5 optimization implementation.

Private Core adapters require their reviewed source contracts. Selection
does not import kernels, load libraries or initialize an accelerator. Remove
this profile when equivalent upstream backend hooks replace the adapters.
"""

import ast
import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path

CORE_WORKER_PATH = "distributed/kv_transfer/kv_connector/v1/hisparse/worker.py"
CORE_CONNECTOR_PATH = "distributed/kv_transfer/kv_connector/v1/hisparse/connector.py"
CORE_SOURCES = {
    "v1/hisparse/runtime.py": "89b940424066b62bf042bcdade3b7efacf03968d9f449888c12b351c636cb926",
    "v1/attention/backends/mla/index_group.py": "dd3dfe9941e1e932f3d7b0c19bcbc177583e7cd6f54295a658eb9be094368cf4",
    CORE_WORKER_PATH: "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229",
    "v1/hisparse/coordinator.py": "530a488c60c3411e53eab42a8fe0624454de942204da304391a4f1c295b30666",
    CORE_CONNECTOR_PATH: "78610e02ebdd622dc8272a98d75c2a75a41d8b19ce1e72ac582bdedd19d11f13",
}
MAIN_CORE_SOURCES = {
    "v1/hisparse/runtime.py": "29659727cd515cb6ed496e254e6bbe174d33ed61f6479e96d61d47dac6afc0dd",
    "v1/attention/backends/mla/index_group.py": "0a25c4558357f568144ac17d74c9fc796b576bf5c94817c95623c36fc9b860ba",
    CORE_WORKER_PATH: "e07bfb44260b30a506208fd180b7530584338f9d6156acbc788ed794d3bd18f4",
    "v1/hisparse/coordinator.py": "141edbeb85bb051ad296970ccc67c48e1f49d85a0cc25e196d59ed775e46dcf9",
    CORE_CONNECTOR_PATH: "b49518f82b1976919a3f50f21430279810473d1c657629b693d121ab3b9a76e7",
}
MAIN_CORE_AST_DIGESTS = {
    "v1/hisparse/runtime.py": "67e404da122d1308e6ecf97559075aa0784026c0119a34f3ecab155b37891f5b",
    "v1/attention/backends/mla/index_group.py": "22f312e38d713729a04a3f978166113684e0ab3e716e187f3c5c6ff6c91d4514",
    CORE_WORKER_PATH: "646f99c78f8003f5dbbc51e5c2996ddd251d1d3afa97c6b03d397381e6301504",
    "v1/hisparse/coordinator.py": "8440c938b7341525bd2ab563a2ec9fb046aaffa3b3aa5ff27de1573c23c673f4",
    CORE_CONNECTOR_PATH: "e9657adbab30213b980b9a659e377d63d3e8208ad2c09357381b173b882f7bf1",
}


def source_contract_digest(source: bytes) -> str:
    """Fingerprint executable syntax without comments/formatting/docstrings.

    Private hooks still require reviewed behavior. Unrelated textual edits
    must not make the same implementation unusable. Preserve signatures,
    decorators, annotations, nonempty type parameters and every statement.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
        if isinstance(node, ast.Module):
            node.type_ignores = []
        if "type_params" in node._fields and getattr(node, "type_params", None) == []:
            node._fields = tuple(name for name in node._fields if name != "type_params")
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _matches_core_source(source: bytes, relative: str) -> bool:
    source = source.replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() == MAIN_CORE_SOURCES[relative]:
        return True
    expected = MAIN_CORE_AST_DIGESTS.get(relative)
    if expected is None:
        return False
    try:
        return source_contract_digest(source) == expected
    except (SyntaxError, UnicodeError):
        return False


def validate_core_sources(package: Path, profile: str = "a5_c8_main") -> None:
    """Reject a partial/different pairing before any common module is patched."""
    if profile != "a5_c8_main":
        raise ValueError("HiSparse optimization requires the reviewed merged-main profile")
    for relative in MAIN_CORE_SOURCES:
        try:
            source = (package / relative).read_bytes()
        except OSError as error:
            raise RuntimeError(f"A5 HiSparse optimization requires its paired Core source: {relative}") from error
        if not _matches_core_source(source, relative):
            raise RuntimeError(f"Re-review A5 HiSparse optimization for the current Core file: {relative}")


def verify_core_source(path: Path, legacy_digest: str) -> None:
    """Accept one reviewed variant, never a mixture of current-main files.

    Historical adapter digests identify their corresponding Core file. They
    never permit the pre-merge implementation to bypass the current source set.
    """
    relative = next((name for name, digest in CORE_SOURCES.items() if digest == legacy_digest), None)
    if relative is None or not _matches_core_source(path.read_bytes(), relative):
        raise RuntimeError(f"Unreviewed HiSparse source: {path.name}")
    package = path
    for _ in Path(relative).parts:
        package = package.parent
    validate_core_sources(package, "a5_c8_main")


def select_profile(requested: str, package: Path | None = None) -> str:
    if requested == "baseline":
        return requested
    if requested != "a5_c8_main":
        raise ValueError("VLLM_ASCEND_HISPARSE_PROFILE must be baseline or a5_c8_main")
    if package is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("A5 HiSparse optimization requires an installed paired vLLM package")
        package = Path(spec.origin).parent
    validate_core_sources(package, requested)
    return requested


@lru_cache(maxsize=1)
def optimized_profile_enabled() -> bool:
    # Configuration/source are immutable for a worker lifetime. Restart after
    # changing either; switching captured graph owners is unsupported.
    from vllm_ascend import envs

    return select_profile(envs.VLLM_ASCEND_HISPARSE_PROFILE) != "baseline"


def backend_class_path() -> str:
    module = "backend_optimized" if optimized_profile_enabled() else "backend"
    return f"vllm_ascend.distributed.kv_transfer.hisparse.{module}.AscendHiSparseBackend"
