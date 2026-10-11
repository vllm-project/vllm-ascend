# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Import-time backport of vLLM #56227 without changing installed vLLM files.

Activate by prepending this directory to PYTHONPATH, before starting Python.
The bundled diff is #56227 applied to v0.30.0; scheduler initialization retains
v0.30.0's lookahead initialization. Remove this adapter after upgrading the pin.
"""

import importlib.abc
import importlib.machinery
import importlib.metadata
import linecache
import re
import sys
from pathlib import Path


def read_patches(path: Path) -> dict[str, tuple[str, ...]]:
    patches = {}
    module = None
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("diff --git "):
            filename = line.split()[3].removeprefix("b/")
            module = filename.removesuffix(".py").replace("/", ".")
            patches[module] = []
        elif module is not None:
            patches[module].append(line)
    return {name: tuple(lines) for name, lines in patches.items()}


def apply_patch(source: str, patch: tuple[str, ...], module: str) -> str:
    """Apply exact-position hunks, rejecting mismatched image source."""
    original = source.splitlines(keepends=True)
    output = []
    cursor = 0
    index = 0
    while index < len(patch):
        header = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", patch[index])
        if header is None:
            index += 1
            continue
        start = int(header[1]) - 1
        old_count = int(header[2] or 1)
        new_count = int(header[4] or 1)
        if start < cursor:
            raise ImportError(f"Overlapping bounded replay hunks in {module}")
        output.extend(original[cursor:start])
        before, after = [], []
        index += 1
        while index < len(patch) and not patch[index].startswith("@@ "):
            line = patch[index]
            if line.startswith(" "):
                before.append(line[1:])
                after.append(line[1:])
            elif line.startswith("-"):
                before.append(line[1:])
            elif line.startswith("+"):
                after.append(line[1:])
            else:
                break
            index += 1
        if len(before) != old_count or len(after) != new_count:
            raise ImportError(f"Invalid bounded replay hunk in {module}")
        if original[start : start + old_count] != before:
            raise ImportError(f"vLLM 0.30.0 source mismatch in {module} at line {start + 1}")
        output.extend(after)
        cursor = start + old_count
    output.extend(original[cursor:])
    return "".join(output)


class ReplaySourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name: str, path: str, patch: tuple[str, ...]):
        super().__init__(name, path)
        self.patch = patch

    def get_source(self, fullname: str) -> str:
        source = super().get_source(fullname)
        if source is None:
            raise ImportError(f"Missing vLLM source for {fullname}")
        return apply_patch(source, self.patch, fullname)

    def get_code(self, fullname: str):
        # Never read or write bytecode caches: workers must see the same diff,
        # and the image's source and .pyc files must remain unchanged.
        source = self.get_source(fullname)
        # inspect.getsource (including CLI field documentation) and tracebacks
        # must use the compiled source rather than the unchanged image file.
        linecache.cache[self.path] = (len(source), None, source.splitlines(keepends=True), self.path)
        return self.source_to_code(source, self.path)


class ReplayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, patches: dict[str, tuple[str, ...]]):
        self.patches = patches

    def find_spec(self, fullname, path=None, target=None):
        patch = self.patches.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ImportError(f"Bounded replay requires Python source for {fullname}")
        spec.loader = ReplaySourceLoader(fullname, spec.origin, patch)
        return spec


def install() -> None:
    if any(isinstance(finder, ReplayFinder) for finder in sys.meta_path):
        return
    version = importlib.metadata.version("vllm").split("+")[0]
    if version != "0.30.0":
        raise ImportError(f"This bounded replay adapter requires vLLM 0.30.0, found {version}")
    patches = read_patches(Path(__file__).with_name("bounded_replay.patch"))
    loaded = patches.keys() & sys.modules.keys()
    if loaded:
        raise ImportError(f"Activate bounded replay before importing vLLM: {sorted(loaded)}")
    sys.meta_path.insert(0, ReplayFinder(patches))
