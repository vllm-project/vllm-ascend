"""Symbolic Docker ENV values, evaluated against the inherited environment."""

import shlex

import regex as re

_VARIABLE = re.compile(r"(?<!\\)(?:\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([-+])([^{}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*))")
_ESCAPED_DOLLAR = "__BISECT_LITERAL_DOLLAR__"


def environment_assignments(body: str) -> tuple[tuple[str, str], ...]:
    if _ESCAPED_DOLLAR in body:
        raise ValueError("ENV contains a reserved parser escape token")
    words = [word.replace(_ESCAPED_DOLLAR, r"\$") for word in shlex.split(body.replace(r"\$", _ESCAPED_DOLLAR))]
    if words and "=" not in words[0]:
        words = [words[0] + "=" + " ".join(words[1:])]
    assignments = []
    for word in words:
        name, separator, value = word.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(f"unsupported ENV assignment: {word!r}")
        if re.search(r"(?<!\\)\$\{[^}]*\$\{", value) or any(
            _VARIABLE.fullmatch(expression) is None for expression in re.findall(r"(?<!\\)\$\{[^}]*\}", value)
        ):
            raise ValueError(f"unsupported ENV expansion: {value!r}")
        assignments.append((name, value))
    return tuple(assignments)


def expand_environment(value: str, inherited: dict[str, str]) -> str:
    def substitute(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(4)
        current = inherited.get(name, "")
        if match.group(2) == "-":
            return current or match.group(3)
        if match.group(2) == "+":
            return match.group(3) if current else ""
        return current

    return _VARIABLE.sub(substitute, value).replace(r"\$", "$")


def environment_references(value: str) -> set[str]:
    return {match.group(1) or match.group(4) for match in _VARIABLE.finditer(value)}


def build_environment(expressions: tuple[tuple[str, str], ...], inherited: dict[str, str]) -> dict[str, str]:
    environment = dict(inherited)
    environment.update((name, expand_environment(value, inherited)) for name, value in expressions)
    return environment


def recover_inherited_environment(
    previous: tuple[tuple[str, str], ...], current: dict[str, str]
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Undo a known prefix/suffix transform; never guess an overwritten value."""
    inherited = dict(current)
    unknown = []
    for name, expression in previous:
        matches = list(_VARIABLE.finditer(expression))
        if len(matches) == 1 and (matches[0].group(1) or matches[0].group(4)) == name and not matches[0].group(2):
            match = matches[0]
            prefix, suffix = expression[: match.start()], expression[match.end() :]
            actual = current.get(name, "")
            if actual.startswith(prefix) and actual.endswith(suffix) and len(actual) >= len(prefix) + len(suffix):
                inherited[name] = actual[len(prefix) : len(actual) - len(suffix) if suffix else None]
                continue
        unknown.append(name)
    return inherited, tuple(unknown)
