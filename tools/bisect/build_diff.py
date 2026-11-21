"""Structural comparison of two ordered Docker build instruction trees."""

import difflib
from dataclasses import dataclass

from tools.bisect.build_recipe import BuildStep


@dataclass(frozen=True)
class BuildChange:
    kind: str
    old_index: int | None
    new_index: int | None
    old: BuildStep | None
    new: BuildStep | None
    argument: str | None = None


def compare_build_chains(old_steps: tuple[BuildStep, ...], new_steps: tuple[BuildStep, ...]) -> list[BuildChange]:
    """Align instructions by syntax, then compare the ARG edges of matched nodes."""
    old_keys = [(step.instruction, step.command) for step in old_steps]
    new_keys = [(step.instruction, step.command) for step in new_steps]
    matcher = difflib.SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
    changes = []
    for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if operation == "equal":
            for old_index, new_index in zip(range(old_start, old_end), range(new_start, new_end), strict=True):
                old, new = old_steps[old_index], new_steps[new_index]
                before, after = dict(old.inputs), dict(new.inputs)
                for argument in sorted(before.keys() | after.keys()):
                    if before.get(argument) != after.get(argument):
                        changes.append(BuildChange("input", old_index, new_index, old, new, argument))
            continue
        if operation == "replace":
            pairs = min(old_end - old_start, new_end - new_start)
            for offset in range(pairs):
                old_index, new_index = old_start + offset, new_start + offset
                changes.append(BuildChange("command", old_index, new_index, old_steps[old_index], new_steps[new_index]))
            old_start += pairs
            new_start += pairs
        for old_index in range(old_start, old_end):
            changes.append(BuildChange("removed", old_index, None, old_steps[old_index], None))
        for new_index in range(new_start, new_end):
            changes.append(BuildChange("added", None, new_index, None, new_steps[new_index]))
    return changes


def command_fragments(old: str, new: str) -> list[str]:
    """Show the changed shell tokens in one RUN instruction."""
    before, after = old.split(), new.split()
    return [
        f"-{' '.join(before[i:j])[:80]} +{' '.join(after[new_start:new_end])[:80]}"
        for operation, i, j, new_start, new_end in difflib.SequenceMatcher(
            a=before, b=after, autojunk=False
        ).get_opcodes()
        if operation != "equal"
    ]
