# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Keep image cleanup commands from deleting a live replay's runtime files."""

import fnmatch
import posixpath
import shlex
from pathlib import PurePosixPath

from tools.bisect.build_analyzer.ir import Instruction
from tools.bisect.build_analyzer.syntax import parse_shell

_TEMPORARY_ROOTS = frozenset({"/tmp", "/var/tmp"})
_SHELLS = frozenset({"bash", "sh", "dash"})


def protect_replay_cleanup(command: str, protected_paths: tuple[str, ...] = ()) -> tuple[str, tuple[str, ...]]:
    """Rewrite only rm operands, keeping shell control flow and other effects.

    Global temporary cleanup is an image-size optimization, not a replay
    requirement. Explicit runtime paths are additionally protected at execution.
    Paths and nested shell scripts must be literal; no shell expansion is run
    by this analysis.
    """
    tree, data = parse_shell(Instruction("<replay-cleanup>", 1, "RUN", command))
    edits: list[tuple[int, int, bytes]] = []
    warnings: list[str] = []

    def literal(node) -> str | None:
        if any(child.type in {"expansion", "simple_expansion", "command_substitution"} for child in descendants(node)):
            return None
        words = shlex.split(data[node.start_byte : node.end_byte].decode("utf-8"))
        return words[0] if len(words) == 1 else None

    def descendants(node):
        yield node
        for child in node.named_children:
            yield from descendants(child)

    def visit(node) -> None:
        if node.type != "command":
            for child in node.named_children:
                visit(child)
            return
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = literal(name_node)
        arguments = node.children_by_field_name("argument")
        if name is not None and PurePosixPath(name).name == "rm":
            removed = []
            local_edits = []
            options_finished = False
            operands = 0
            for argument in arguments:
                value = literal(argument)
                if value == "--" and not options_finished:
                    options_finished = True
                    continue
                if value is not None and value.startswith("-") and not options_finished:
                    continue
                operands += 1
                if value is None or not _protected_target(value, protected_paths):
                    continue
                removed.append(value)
                local_edits.append((argument.start_byte - node.start_byte, argument.end_byte - node.start_byte))
            if removed:
                text = data[node.start_byte : node.end_byte]
                for start, end in reversed(local_edits):
                    text = text[:start] + text[end:]
                message = "[replay-cleanup:warning] skipped removal of protected cleanup targets: " + ", ".join(removed)
                warnings.append(message)
                notify = "printf '%s\\n' " + shlex.quote(message) + " >&2"
                # A successful no-op preserves && chains when all operands go.
                replacement = "{ " + notify + "; "
                replacement += (text.decode("utf-8") + "; " if operands > len(removed) else "") + "}"
                edits.append((node.start_byte, node.end_byte, replacement.encode("utf-8")))
            return
        # Recovered image RUNs are wrapped in env ... /bin/bash -c '...'.
        # Rewrite the literal script rather than matching text inside quotes.
        if name is not None and PurePosixPath(name).name in (_SHELLS | {"env"}):
            words = [literal(argument) for argument in arguments]
            shell_seen = PurePosixPath(name).name in _SHELLS
            for index, value in enumerate(words):
                if value and PurePosixPath(value).name in _SHELLS:
                    shell_seen = True
                if shell_seen and value in {"-c", "-lc", "-cl"} and index + 1 < len(arguments):
                    script = arguments[index + 1]
                    # Single quotes have no outer-shell expansion to preserve.
                    if script.type == "raw_string" and words[index + 1] is not None:
                        rewritten, nested_warnings = protect_replay_cleanup(words[index + 1], protected_paths)
                        if nested_warnings:
                            edits.append((script.start_byte, script.end_byte, shlex.quote(rewritten).encode("utf-8")))
                            warnings.extend(nested_warnings)
                    break
        for child in node.named_children:
            if child.type == "command_substitution":
                visit(child)

    visit(tree.root_node)
    for start, end, replacement in sorted(edits, reverse=True):
        data = data[:start] + replacement + data[end:]
    return data.decode("utf-8"), tuple(warnings)


def _protected_target(value: str, protected_paths: tuple[str, ...]) -> bool:
    target = posixpath.normpath(value)
    if target == "/" or target in _TEMPORARY_ROOTS:
        return True
    if posixpath.dirname(target) in _TEMPORARY_ROOTS and posixpath.basename(target) in {"*", ".*", "**"}:
        return True
    for protected in protected_paths:
        path = PurePosixPath(protected)
        if not path.is_absolute():
            continue
        if any(fnmatch.fnmatchcase(str(ancestor), target) for ancestor in (path, *path.parents)):
            return True
        if target.startswith(str(path).rstrip("/") + "/"):
            return True
    return False
