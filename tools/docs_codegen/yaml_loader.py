from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.docs_codegen.errors import make_docs_codegen_error
from tools.docs_codegen.scanner import ModelCodeBlock


@dataclass(frozen=True)
class LoadedYaml:
    """One loaded YAML document referenced by a ``model-code`` block."""

    yaml_path: Path
    yaml_root: Any


class YamlLoader:
    """Load and cache one repository-relative YAML file."""

    def __init__(self) -> None:
        self._yaml_cache: dict[Path, Any] = {}

    def load(
        self,
        *,
        test_case_path: str,
        block: ModelCodeBlock | None = None,
    ) -> LoadedYaml:
        yaml_path = self._resolve_test_case_path(test_case_path=test_case_path, block=block)
        yaml_root = self._load_yaml_root(yaml_path)
        return LoadedYaml(
            yaml_path=yaml_path,
            yaml_root=yaml_root,
        )

    def _resolve_test_case_path(self, *, test_case_path: str, block: ModelCodeBlock | None = None) -> Path:
        candidate = Path(test_case_path)
        if candidate.is_absolute():
            raise make_docs_codegen_error(
                "test_case_path must be a repository-relative path",
                block=block,
                test_case_path=test_case_path,
            )

        if not candidate.exists():
            raise make_docs_codegen_error(
                "test_case_path file does not exist",
                block=block,
                test_case_path=test_case_path,
            )
        return candidate.resolve()

    @staticmethod
    def _load_yaml_file(yaml_path: Path) -> Any:
        with yaml_path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _load_yaml_root(self, yaml_path: Path) -> Any:
        if yaml_path not in self._yaml_cache:
            self._yaml_cache[yaml_path] = self._load_yaml_file(yaml_path)
        return self._yaml_cache[yaml_path]
