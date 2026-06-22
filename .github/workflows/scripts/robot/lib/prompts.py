"""System prompt loading for robot workflows.

Supports three prompt variants: ``issue``, ``pr``, and ``commit``.
"""

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "description_check_prompts"
COMMIT_PROMPT_DIR = Path(__file__).resolve().parent.parent / "commit_check_prompts"
DEFAULT_FALLBACK = "You are an issue triage assistant."

PROMPT_NAME_MAP: dict[str, tuple[str, str]] = {
    "issue": ("description_check_prompts", "system_prompt.txt"),
    "pr": ("description_check_prompts", "pr_system_prompt.txt"),
    "commit": ("commit_check_prompts", "system_prompt.txt"),
}


def load_system_prompt(variant: str = "issue") -> str:
    """Load a system prompt text file for the given *variant*.

    Args:
        variant: One of ``"issue"``, ``"pr"``, or ``"commit"``.

    Returns:
        The prompt text, or a fallback message if the file is missing.
    """
    if variant in PROMPT_NAME_MAP:
        subdir, filename = PROMPT_NAME_MAP[variant]
        prompt_dir = COMMIT_PROMPT_DIR if variant == "commit" else PROMPT_DIR
        path = prompt_dir / filename
    else:
        path = PROMPT_DIR / "system_prompt.txt"

    if path.exists():
        return path.read_text()
    return DEFAULT_FALLBACK
