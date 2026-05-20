#!/usr/bin/env python3
"""
Generate release checklist issue body from template.

Usage:
    python generate_checklist.py \
        --version v0.15.0rc1 \
        --branch main \
        --date 2026.03.15 \
        --manager wangxiyuan \
        --feedback-issue 1234 \
        --output release-checklist.md
"""

import argparse
from pathlib import Path


def load_template(template_path: Path) -> str:
    """Load the release checklist template."""
    with open(template_path) as f:
        return f.read()


def substitute_variables(template: str, variables: dict) -> str:
    """Substitute variables in the template."""
    result = template
    for key, value in variables.items():
        result = result.replace(f"${{{key}}}", value)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate release checklist from template")
    parser.add_argument("--version", required=True, help="Release version (e.g., v0.15.0rc1)")
    parser.add_argument("--branch", default="main", help="Release branch")
    parser.add_argument("--date", required=True, help="Target release date (e.g., 2026.03.15)")
    parser.add_argument("--manager", required=True, help="Release manager GitHub username")
    parser.add_argument("--feedback-issue", type=int, help="Feedback issue number")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--template",
        default=None,
        help="Template file path (default: templates/release-checklist-template.md)",
    )

    args = parser.parse_args()

    # Determine template path
    script_dir = Path(__file__).parent
    if args.template:
        template_path = Path(args.template)
    else:
        template_path = script_dir.parent / "templates" / "release-checklist-template.md"

    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        return 1

    # Load template
    template = load_template(template_path)

    # Build feedback issue URL
    feedback_issue_url = ""
    if args.feedback_issue:
        feedback_issue_url = f"https://github.com/vllm-project/vllm-ascend/issues/{args.feedback_issue}"

    # Prepare variables
    variables = {
        "VERSION": args.version,
        "BRANCH": args.branch,
        "DATE": args.date,
        "MANAGER": args.manager,
        "FEEDBACK_ISSUE_URL": feedback_issue_url,
        "RELEASE_NOTE_PR_URL": "",  # To be filled later
        "BUG_LIST": "<!-- Run scan_release_bugs.py to populate -->",
        "PR_LIST": "<!-- Run identify_release_prs.py to populate -->",
        "FUNCTIONAL_TEST_RESULTS": "<!-- Run run_functional_tests.py to populate -->",
    }

    # Generate checklist
    checklist = substitute_variables(template, variables)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(checklist)

    print(f"Generated release checklist at {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
