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

"""Bash syntactic analysis with source-positioned errors."""

from dataclasses import dataclass

import regex as re

from tools.bisect.build_analyzer.ir import BuildSyntaxError, Instruction
from tools.bisect.build_analyzer.lexer import lex_dockerfile

DOCKER_INSTRUCTIONS = frozenset(
    {
        "ADD",
        "ARG",
        "CMD",
        "COPY",
        "ENTRYPOINT",
        "ENV",
        "EXPOSE",
        "FROM",
        "HEALTHCHECK",
        "LABEL",
        "MAINTAINER",
        "ONBUILD",
        "RUN",
        "SHELL",
        "STOPSIGNAL",
        "USER",
        "VOLUME",
        "WORKDIR",
    }
)


def parse_dockerfile(source: str, content: str) -> tuple[Instruction, ...]:
    """Parse logical Dockerfile tokens into typed instructions."""
    instructions: list[Instruction] = []
    for token in lex_dockerfile(source, content):
        match = re.fullmatch(r"([A-Za-z]+)\s+(.*)", token.text, re.DOTALL)
        if match is None:
            raise BuildSyntaxError(f"{source}:{token.line}: invalid Dockerfile instruction")
        kind = match.group(1).upper()
        if kind not in DOCKER_INSTRUCTIONS:
            raise BuildSyntaxError(f"{source}:{token.line}: unsupported Dockerfile instruction {kind}")
        body = match.group(2)
        if kind != "RUN":
            body = re.sub(r"\\\n\s*", " ", body)
        instructions.append(Instruction(source, token.line, kind, body, token.line_map))
    return tuple(instructions)


def parse_shell(instruction: Instruction):
    """Parse and validate a RUN shell program; return the AST and source bytes."""
    data = instruction.body.encode("utf-8")
    tree = _bash_parser().parse(data)
    errors = list(_errors(tree.root_node))
    if errors:
        first = errors[0]
        row = first.start_point.row
        line = instruction.line_map[row] if row < len(instruction.line_map) else instruction.line + row
        raise BuildSyntaxError(
            f"{instruction.source}:{line}: Bash syntax error at column {first.start_point.column + 1}"
        )
    return tree, data


@dataclass(frozen=True)
class SyntaxUnit:
    instruction: Instruction
    shell_tree: object | None = None
    shell_source: bytes | None = None


def parse_build(layers: tuple[tuple[str, str], ...]) -> tuple[tuple[SyntaxUnit, ...], ...]:
    """Produce a complete syntax tree for each Docker build layer."""
    parsed_layers: list[tuple[SyntaxUnit, ...]] = []
    for source, content in layers:
        units: list[SyntaxUnit] = []
        for instruction in parse_dockerfile(source, content):
            body = instruction.body.lstrip()
            if instruction.kind == "RUN" and not body.startswith(("[", "--")):
                tree, data = parse_shell(instruction)
                units.append(SyntaxUnit(instruction, tree, data))
            else:
                units.append(SyntaxUnit(instruction))
        parsed_layers.append(tuple(units))
    return tuple(parsed_layers)


def _bash_parser():
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:
        raise BuildSyntaxError("Bash parser missing; install tree-sitter-language-pack==0.13.0") from exc
    return get_parser("bash")


def _node_text(source: bytes, node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _canonical_syntax(source: bytes, node) -> str:
    """Serialize syntax tokens without source whitespace or comments."""
    tokens: list[str] = []

    def collect(current) -> None:
        if current.type == "comment":
            return
        if not current.children:
            tokens.append(_node_text(source, current))
            return
        for child in current.children:
            collect(child)

    collect(node)
    return " ".join(tokens)


def _errors(node):
    if node.type == "ERROR" or node.is_missing:
        yield node
    for child in node.children:
        yield from _errors(child)
