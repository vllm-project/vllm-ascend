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

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else None
        self._yaml_cache: dict[Path, Any] = {}

    # -- Public API ----------------------------------------------------------

    def load(
        self,
        *,
        test_case_path: str,
        block: ModelCodeBlock | None = None,
    ) -> LoadedYaml:
        """Resolve, parse, and cache the YAML referenced by a model-code block."""
        yaml_path = self._resolve_test_case_path(test_case_path=test_case_path, block=block)
        yaml_root = self._load_yaml_root(yaml_path)
        return LoadedYaml(
            yaml_path=yaml_path,
            yaml_root=yaml_root,
        )

    # -- Resolution & parsing ------------------------------------------------

    def _resolve_test_case_path(self, *, test_case_path: str, block: ModelCodeBlock | None = None) -> Path:
        """Resolve a repo-relative ``test_case_path`` to an absolute, contained, existing file."""
        candidate = Path(test_case_path)
        if candidate.is_absolute():
            raise make_docs_codegen_error(
                "test_case_path must be a repository-relative path",
                block=block,
                test_case_path=test_case_path,
            )

        base = self._base.resolve()
        resolved = (base / candidate).resolve()
        if not resolved.is_relative_to(base):
            raise make_docs_codegen_error(
                "test_case_path must stay within the repository",
                block=block,
                test_case_path=test_case_path,
            )

        if not resolved.exists():
            raise make_docs_codegen_error(
                "test_case_path file does not exist",
                block=block,
                test_case_path=test_case_path,
            )
        return resolved

    def _load_yaml_root(self, yaml_path: Path) -> Any:
        """Return the parsed YAML for ``yaml_path``, caching it on first load."""
        if yaml_path not in self._yaml_cache:
            self._yaml_cache[yaml_path] = self._parse_yaml_file(yaml_path)
        return self._yaml_cache[yaml_path]

    @staticmethod
    def _parse_yaml_file(yaml_path: Path) -> Any:
        """Read and parse one YAML file, treating an empty document as ``{}``."""
        with yaml_path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    # -- Path helpers --------------------------------------------------------

    @property
    def _base(self) -> Path:
        """Directory that repo-relative paths resolve against for filesystem I/O."""
        return self.repo_root if self.repo_root is not None else Path.cwd()
