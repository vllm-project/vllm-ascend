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

"""Evaluate known Docker/Bash inputs and lower selected effects to IR."""

import fnmatch
import json
import posixpath
import re
import shlex
from urllib.parse import urlsplit

from tools.bisect.build_analyzer.ir import BuildAnalysis, BuildSyntaxError, Instruction, Operation, ShellDependency
from tools.bisect.build_analyzer.package_effects import package_effects, package_outcomes
from tools.bisect.build_analyzer.syntax import SyntaxUnit, _canonical_syntax, _node_text, parse_build

_BRANCHES = frozenset({"if_statement", "case_statement", "for_statement", "while_statement", "until_statement"})
_COMMANDS = frozenset({"command", "declaration_command"})


def _download_output(command: str) -> str | None:
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if not words or words[0] not in ("wget", "curl"):
        return None
    for flag in ("-O", "-o", "--output"):
        if flag in words and words.index(flag) + 1 < len(words):
            return words[words.index(flag) + 1]
    if "-P" in words and words.index("-P") + 1 < len(words):
        url = next((word for word in words if word.startswith(("https://", "http://"))), None)
        if url:
            return posixpath.join(words[words.index("-P") + 1], posixpath.basename(urlsplit(url).path))
    return None


def _download_url(command: str) -> str | None:
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    urls = [word for word in words if word.startswith(("https://", "http://"))]
    return urls[0] if len(urls) == 1 else None


def _references(text: str) -> tuple[str, ...]:
    names = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}|\$([A-Za-z_][A-Za-z0-9_]*)", text)
    return tuple(sorted({first or second for first, second in names}))


def _command_kind(source: bytes, node) -> str:
    if node.type == "declaration_command" and _node_text(source, node).startswith("export "):
        return "export"
    words = [_node_text(source, child) for child in node.named_children if child.type != "variable_assignment"]
    if not words:
        return "shell"
    if words[:3] in (["python", "-m", "pip"], ["python3", "-m", "pip"]):
        words = words[2:]
    executable = words[0]
    if executable in ("wget", "curl"):
        return "download"
    if executable in ("pip", "pip3") and len(words) > 1:
        return f"pip-{words[1]}"
    if executable in ("git", "apt-get", "mfcli") and len(words) > 1:
        return f"{executable}-{words[1]}"
    if executable in ("export", "source", "cd", "bash"):
        return executable
    return "shell"


def _shell_operations(unit: SyntaxUnit, variables: dict[str, str | None]) -> tuple[list[Operation], list[str]]:
    instruction = unit.instruction
    tree, data = unit.shell_tree, unit.shell_source
    if tree is None or data is None:
        raise BuildSyntaxError(f"{instruction.source}:{instruction.line}: RUN syntax tree missing")
    operations: list[Operation] = []
    diagnostics: list[str] = []
    scope = variables.copy()
    shell_dependencies: list[ShellDependency] = []

    def resolve(value: str, scalar: bool = False) -> tuple[str, tuple[str, ...]]:
        value = value.strip()
        if scalar and len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        default = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}", value)
        if default:
            value = scope.get(default.group(1)) or default.group(2)
        return _expand(value, scope)

    def visit(node, condition: str | None = None) -> None:
        row = node.start_point.row
        line = instruction.line_map[row] if row < len(instruction.line_map) else instruction.line + row
        if node.type == "variable_assignment":
            raw = _node_text(data, node)
            name, _, value = raw.partition("=")
            resolved, unknown = resolve(value, scalar=True)
            scope[name] = resolved if not unknown else None
            operations.append(
                Operation(
                    instruction.source,
                    line,
                    "assign",
                    f"{name}={resolved}",
                    condition,
                    unknown,
                    _references(value),
                    tuple(shell_dependencies),
                )
            )
            return
        if node.type == "case_statement":
            selector = node.named_children[0]
            selected, unknown = resolve(_node_text(data, selector), scalar=True)
            items = [child for child in node.named_children if child.type == "case_item"]
            if not unknown:
                for item in items:
                    patterns = []
                    for child in item.named_children:
                        if child.type in ("string", "word", "raw_string", "extglob_pattern"):
                            patterns.append(_node_text(data, child).strip("\"'"))
                        else:
                            break
                    if any(fnmatch.fnmatchcase(selected, pattern) for pattern in patterns):
                        operations.append(
                            Operation(
                                instruction.source,
                                line,
                                "case-select",
                                selected,
                                inputs=_references(_node_text(data, selector)),
                            )
                        )
                        for child in item.named_children[len(patterns) :]:
                            visit(child, condition)
                        return
            diagnostics.append(f"{instruction.source}:{line}: case outcome unknown")
            operations.append(
                Operation(
                    instruction.source,
                    line,
                    "branch",
                    _canonical_syntax(data, node),
                    condition,
                    ("case outcome unknown",),
                    _references(_node_text(data, node)),
                )
            )
            return
        if node.type == "if_statement":
            children = node.named_children
            predicate = children[0] if children else None
            expression = _node_text(data, predicate) if predicate is not None else ""
            match = re.fullmatch(r"\[\[?\s+-n\s+(.+?)\s+\]\]?", expression)
            if match and not any(child.type == "elif_clause" for child in children):
                value, unknown = resolve(match.group(1), scalar=True)
                if not unknown:
                    passed = bool(value)
                    operations.append(
                        Operation(
                            instruction.source, line, "if-select", f"nonempty={passed}", inputs=_references(expression)
                        )
                    )
                    selected = [child for child in children[1:] if (child.type != "else_clause") == passed]
                    for child in selected:
                        visit(child, condition)
                    return
            branches = [child for child in children[1:] if child.type in ("command", "else_clause")]
            if (
                len(branches) == 2
                and branches[1].type == "else_clause"
                and not any(child.type == "elif_clause" for child in children)
                and len(branches[1].named_children) == 1
            ):
                branch_commands = [
                    branch
                    if branch.type == "command"
                    else next((child for child in branch.named_children if child.type == "command"), None)
                    for branch in branches
                ]
                if all(command is not None for command in branch_commands):
                    commands = [resolve(_node_text(data, command))[0] for command in branch_commands]
                    outputs = [_download_output(command) for command in commands]
                    urls = [_download_url(command) for command in commands]
                    if (
                        outputs[0] is not None
                        and outputs[0] == outputs[1]
                        and urls[0] is not None
                        and urls[0] == urls[1]
                    ):
                        resolved, _ = resolve(_node_text(data, node))
                        operations.append(
                            Operation(
                                instruction.source,
                                line,
                                "download",
                                resolved,
                                inputs=_references(_node_text(data, node)),
                                shell_dependencies=tuple(shell_dependencies),
                                outputs=(outputs[0],),
                            )
                        )
                        return
            diagnostics.append(f"{instruction.source}:{line}: if outcome unknown")
            operations.append(
                Operation(
                    instruction.source,
                    line,
                    "branch",
                    _canonical_syntax(data, node),
                    condition,
                    ("if outcome unknown",),
                    _references(_node_text(data, node)),
                )
            )
            return
        if node.type in _BRANCHES:
            branch = _canonical_syntax(data, node)
            marker = f"{instruction.source}:{line}: {node.type}"
            diagnostics.append(f"{marker} requires branch evaluation")
            operations.append(
                Operation(
                    instruction.source,
                    line,
                    "branch",
                    branch,
                    condition,
                    ("branch outcome unknown",),
                    _references(_node_text(data, node)),
                )
            )
            return
        if node.type in _COMMANDS:
            # Keep declaration keywords such as `export`; named children omit them.
            text = _node_text(data, node)
            resolved, unknown = resolve(text)
            kind = _command_kind(data, node)
            operations.append(
                Operation(
                    instruction.source,
                    line,
                    kind,
                    resolved,
                    condition,
                    unknown,
                    _references(text),
                    tuple(shell_dependencies),
                    (_download_output(resolved),) if kind == "download" and _download_output(resolved) else (),
                )
            )
            if kind in ("source", "export", "cd"):
                shell_dependencies.append(ShellDependency(kind, resolved))
            if kind == "export":
                for assignment in re.findall(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", resolved):
                    scope[assignment[0]] = assignment[1].strip("\"'")
            if unknown:
                diagnostics.append(f"{instruction.source}:{line}: unresolved command inputs: {', '.join(unknown)}")
            return
        for child in node.named_children:
            visit(child, condition)

    visit(tree.root_node)
    return operations, diagnostics


def _assignment(body: str, source: str, line: int) -> tuple[str, str | None]:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?$", body.strip(), re.DOTALL)
    if not match:
        raise BuildSyntaxError(f"{source}:{line}: invalid ARG {body!r}")
    return match.group(1), match.group(2)


def _expand(text: str, variables: dict[str, str | None]) -> tuple[str, tuple[str, ...]]:
    unknown: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        value = variables.get(name)
        if value is None:
            unknown.add(name)
            return match.group(0)
        return value

    expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace, text)
    return expanded, tuple(sorted(unknown))


def analyze_build(
    layers: tuple[tuple[str, str], ...],
    build_args: dict[str, str] | None = None,
) -> BuildAnalysis:
    """Analyze ordered root Dockerfile and derived layers for one target.

    Build args are explicit inputs. Without an override, an ARG default is used;
    an ARG without a default is unknown. FROM starts a fresh stage scope while
    preserving global ARGs. RUN shell variables are not assumed to survive RUNs.
    """
    overrides = build_args or {}
    instructions: list[Instruction] = []
    operations: list[Operation] = []
    diagnostics: list[str] = []
    syntax_layers = parse_build(layers)
    for units in syntax_layers:
        global_args: dict[str, str | None] = {}
        stage: dict[str, str | None] = {}
        in_stage = False
        for unit in units:
            instruction = unit.instruction
            source = instruction.source
            instructions.append(instruction)
            kind, body = instruction.kind, instruction.body
            if kind == "ARG":
                name, default = _assignment(body, source, instruction.line)
                value = overrides.get(name, default.strip("\"'") if default is not None else None)
                if in_stage and default is None:
                    value = overrides.get(name, global_args.get(name))
                (stage if in_stage else global_args)[name] = value
                operations.append(
                    Operation(
                        source,
                        instruction.line,
                        "arg",
                        f"{name}={value if value is not None else '<unknown>'}",
                        unresolved=(name,) if value is None else (),
                    )
                )
            elif kind == "FROM":
                resolved, unknown = _expand(body, global_args)
                operations.append(
                    Operation(source, instruction.line, "from", resolved, unresolved=unknown, inputs=_references(body))
                )
                if unknown:
                    diagnostics.append(f"{source}:{instruction.line}: unknown FROM inputs: {', '.join(unknown)}")
                in_stage = True
                stage = {}
            elif kind == "ENV":
                # Dockerfile's key=value form; preserve nontrivial expressions
                # as a raw effect rather than guessing shell semantics.
                resolved, unknown = _expand(body, stage)
                operations.append(
                    Operation(source, instruction.line, "env", resolved, unresolved=unknown, inputs=_references(body))
                )
                for match in re.finditer(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", resolved):
                    stage[match.group(1)] = match.group(2).strip('"')
            elif kind == "RUN":
                if body.lstrip().startswith("["):
                    try:
                        args = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise BuildSyntaxError(f"{source}:{instruction.line}: invalid exec-form RUN") from exc
                    resolved, unknown = _expand(json.dumps(args), stage)
                    operations.append(Operation(source, instruction.line, "exec", resolved, unresolved=unknown))
                elif body.lstrip().startswith("--"):
                    operations.append(
                        Operation(
                            source,
                            instruction.line,
                            "run-option",
                            body,
                            unresolved=("RUN flags require evaluation",),
                        )
                    )
                    diagnostics.append(f"{source}:{instruction.line}: RUN flags require evaluation")
                else:
                    parsed, notes = _shell_operations(unit, stage)
                    operations.extend(parsed)
                    diagnostics.extend(notes)
            else:
                resolved, unknown = _expand(body, stage)
                operations.append(Operation(source, instruction.line, kind.lower(), resolved, unresolved=unknown))
                if unknown:
                    diagnostics.append(f"{source}:{instruction.line}: unresolved {kind} inputs: {', '.join(unknown)}")
    parsed_operations = tuple(operations)
    return BuildAnalysis(
        tuple(instructions),
        parsed_operations,
        tuple(diagnostics),
        package_effects(parsed_operations),
        package_outcomes(parsed_operations),
    )
