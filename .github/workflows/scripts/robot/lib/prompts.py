"""System prompt loading for robot workflows.

Supports two prompt variants: ``issue`` and ``pr``.
"""

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "description_check_prompts"
DEFAULT_FALLBACK = "You are an issue triage assistant."

PROMPT_NAME_MAP: dict[str, str] = {
    "issue": "system_prompt.txt",
    "pr": "pr_system_prompt.txt",
}


def load_system_prompt(variant: str = "issue") -> str:
    """Load a system prompt text file for the given *variant*.

    Args:
        variant: One of ``"issue"`` or ``"pr"``.

    Returns:
        The prompt text, or a fallback message if the file is missing.
    """
    filename = PROMPT_NAME_MAP.get(variant, "system_prompt.txt")
    path = PROMPT_DIR / filename
    if path.exists():
        return path.read_text()
    return DEFAULT_FALLBACK
