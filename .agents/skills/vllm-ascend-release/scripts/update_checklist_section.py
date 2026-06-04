#!/usr/bin/env python3
"""
Update a specific section of the release checklist issue.

This script uses the GitHub CLI to update the body of an issue,
replacing or appending content to a specific section.

Usage:
    python update_checklist_section.py \
        --issue-number 6149 \
        --section "Bug need Solve" \
        --content-file bug-list.md
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def run_gh_command(args: list[str]) -> str:
    """Run a GitHub CLI command and return the output."""
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"gh command failed: {result.stderr}")
    return result.stdout


def get_issue_body(repo: str, issue_number: int) -> str:
    """Get the current body of an issue."""
    output = run_gh_command(
        [
            "issue",
            "view",
            str(issue_number),
            "--repo",
            repo,
            "--json",
            "body",
        ]
    )
    data = json.loads(output)
    return data.get("body", "")


def update_issue_body(repo: str, issue_number: int, new_body: str):
    """Update the body of an issue."""
    run_gh_command(
        [
            "issue",
            "edit",
            str(issue_number),
            "--repo",
            repo,
            "--body",
            new_body,
        ]
    )


def find_section(body: str, section_name: str) -> tuple[int, int]:
    """
    Find the start and end positions of a section in the issue body.

    Returns:
        Tuple of (start_pos, end_pos) where end_pos is the start of the next section
        or end of the body.
    """
    # Match section header (### Section Name)
    section_pattern = rf"^###\s+{re.escape(section_name)}\s*$"
    section_match = re.search(section_pattern, body, re.MULTILINE)

    if not section_match:
        return -1, -1

    start_pos = section_match.end()

    # Find the next section (### followed by any text)
    next_section_pattern = r"^###\s+"
    next_match = re.search(next_section_pattern, body[start_pos:], re.MULTILINE)

    if next_match:
        end_pos = start_pos + next_match.start()
    else:
        end_pos = len(body)

    return start_pos, end_pos


def update_section(body: str, section_name: str, new_content: str, append: bool = False) -> str:
    """
    Update a section in the issue body.

    Args:
        body: The current issue body
        section_name: Name of the section to update
        new_content: New content for the section
        append: If True, append to existing content; if False, replace

    Returns:
        Updated body text
    """
    start_pos, end_pos = find_section(body, section_name)

    if start_pos == -1:
        # Section not found, append at the end
        return body + f"\n\n### {section_name}\n\n{new_content}"

    if append:
        # Get existing content and append
        existing_content = body[start_pos:end_pos].strip()
        new_section_content = f"\n\n{existing_content}\n\n{new_content}\n\n"
    else:
        # Replace content
        new_section_content = f"\n\n{new_content}\n\n"

    return body[:start_pos] + new_section_content + body[end_pos:]


def main():
    parser = argparse.ArgumentParser(description="Update a section of the release checklist issue")
    parser.add_argument("--repo", default="vllm-project/vllm-ascend", help="Repository")
    parser.add_argument("--issue-number", type=int, required=True, help="Issue number")
    parser.add_argument("--section", required=True, help="Section name to update")
    parser.add_argument("--content-file", required=True, help="File containing new content")
    parser.add_argument("--append", action="store_true", help="Append to section instead of replace")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without updating")

    args = parser.parse_args()

    content_path = Path(args.content_file)
    if not content_path.exists():
        print(f"Error: Content file not found: {content_path}")
        return 1

    with open(content_path) as f:
        new_content = f.read().strip()

    print(f"Fetching issue #{args.issue_number}...")
    body = get_issue_body(args.repo, args.issue_number)

    print(f"Updating section '{args.section}'...")
    updated_body = update_section(body, args.section, new_content, args.append)

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print("Updated body would be:")
        print("-" * 40)
        print(updated_body)
        print("-" * 40)
    else:
        update_issue_body(args.repo, args.issue_number, updated_body)
        print(f"Successfully updated section '{args.section}' in issue #{args.issue_number}")

    return 0


if __name__ == "__main__":
    exit(main())
