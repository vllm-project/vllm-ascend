"""Interpret Git global options before identifying a build operation."""

import posixpath
from dataclasses import dataclass


@dataclass(frozen=True)
class GitCommand:
    subcommand: str
    arguments: tuple[str, ...]
    directory: str


def parse_git_command(words: list[str], cwd: str = "/") -> GitCommand | None:
    if not words or words[0] != "git":
        return None
    index = 1
    directory = cwd
    while index < len(words):
        word = words[index]
        if word == "--":
            index += 1
            break
        if not word.startswith("-"):
            break
        if word in {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}:
            if index + 1 >= len(words):
                return None
            if word == "-C":
                directory = posixpath.normpath(posixpath.join(directory, words[index + 1]))
            index += 2
        else:
            if word.startswith("-C"):
                directory = posixpath.normpath(posixpath.join(directory, word[2:]))
            index += 1
    if index >= len(words):
        return None
    return GitCommand(words[index], tuple(words[index + 1 :]), directory)


def git_resources(command: GitCommand) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep repository contents and Git metadata in the dependency graph."""
    arguments = command.arguments
    directory = command.directory
    if command.subcommand == "clone" and len(arguments) >= 2:
        directory = posixpath.normpath(posixpath.join(directory, arguments[-1]))
        return (), (directory.rstrip("/") + "/",)
    if command.subcommand == "init" and arguments and not arguments[-1].startswith("-"):
        directory = posixpath.normpath(posixpath.join(directory, arguments[-1]))
    tree = directory.rstrip("/") + "/"
    metadata = tree + ".git/"
    if command.subcommand == "init":
        return (), (metadata,)
    if command.subcommand == "fetch":
        return (metadata,), (metadata,)
    if command.subcommand in {"checkout", "switch", "reset", "submodule"}:
        return (metadata,), (tree,)
    return (), ()
