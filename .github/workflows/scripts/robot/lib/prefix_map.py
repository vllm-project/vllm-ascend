"""Single source of truth for issue-title prefix mappings.

Consolidates the mappings that were previously duplicated across
``extract_input.py`` and ``call_llm.py``.
"""

TITLE_TO_TEMPLATE: dict[str, str] = {
    "[Bug]": "400-bug-report.yml",
    "[Installation]": "200-installation.yml",
    "[Usage]": "300-usage.yml",
    "[Doc]": "100-documentation.yml",
    "[Misc]": "800-others.yml",
}

PREFIX_TO_TYPE_KEY: dict[str, str] = {
    "[Bug]": "bug",
    "[Installation]": "installation",
    "[Install]": "installation",
    "[Usage]": "usage",
    "[Doc]": "document",
    "[Misc]": "other",
    "[Feature]": "feature",
    "[Perf]": "performance",
}

VALID_TYPE_PREFIXES: list[str] = list(TITLE_TO_TEMPLATE.keys())


def extract_issue_type(title: str) -> str | None:
    """Return the recognised issue-type prefix from *title*, or ``None``.

    The prefix must be followed by ``:`` (e.g. ``[Bug]: description``).

    Args:
        title: The issue or PR title.

    Returns:
        The prefix (e.g. ``"[Bug]"``) or ``None`` if no known prefix matches.
    """
    for prefix in TITLE_TO_TEMPLATE:
        if title.startswith(f"{prefix}:"):
            return prefix
    return None
