#!/usr/bin/env python3
"""
Update version references across documentation files.

This script updates version numbers and compatibility information
in various documentation files for a new release.

Usage:
    python update_version_references.py \
        --version v0.15.0rc1 \
        --vllm-version v0.15.0 \
        --feedback-issue https://github.com/vllm-project/vllm-ascend/issues/1234
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

# Files that need version updates
VERSION_FILES = [
    {
        "path": "README.md",
        "updates": [
            {
                "pattern": r"pip install vllm-ascend==[\d.]+(?:rc\d+)?",
                "replacement": "pip install vllm-ascend=={version}",
            },
            {
                "pattern": r"vllm-ascend:v[\d.]+(?:rc\d+)?",
                "replacement": "vllm-ascend:{version}",
            },
        ],
    },
    {
        "path": "README.zh.md",
        "updates": [
            {
                "pattern": r"pip install vllm-ascend==[\d.]+(?:rc\d+)?",
                "replacement": "pip install vllm-ascend=={version}",
            },
            {
                "pattern": r"vllm-ascend:v[\d.]+(?:rc\d+)?",
                "replacement": "vllm-ascend:{version}",
            },
        ],
    },
    {
        "path": "docs/conf.py",
        "updates": [
            {
                "pattern": r'version = "[\d.]+(?:rc\d+)?"',
                "replacement": 'version = "{version}"',
            },
            {
                "pattern": r'release = "[\d.]+(?:rc\d+)?"',
                "replacement": 'release = "{version}"',
            },
        ],
    },
    {
        "path": "docs/source/faqs.md",
        "updates": [
            {
                "pattern": r"https://github\.com/vllm-project/vllm-ascend/issues/\d+",
                "replacement": "{feedback_issue}",
                "context": "feedback",  # Only update if in feedback context
            },
        ],
    },
]


def update_file(file_path: Path, updates: list[dict], variables: dict, dry_run: bool = False) -> bool:
    """
    Update a single file with version references.

    Returns:
        True if file was modified, False otherwise
    """
    if not file_path.exists():
        print(f"  Warning: File not found: {file_path}")
        return False

    with open(file_path) as f:
        content = f.read()

    modified = False

    for update in updates:
        pattern = update["pattern"]
        replacement = update["replacement"].format(**variables)

        # Check for context restriction
        if "context" in update:
            # Only apply if we can confirm context
            # For now, apply all updates
            pass

        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            content = new_content
            modified = True
            print(f"  Updated: {pattern[:50]}...")

    if modified:
        if dry_run:
            print(f"  [DRY RUN] Would update {file_path}")
        else:
            with open(file_path, "w") as f:
                f.write(content)
            print(f"  Updated: {file_path}")

    return modified


def update_versioning_policy(
    file_path: Path,
    version: str,
    vllm_version: str,
    dry_run: bool = False,
) -> bool:
    """
    Update the versioning policy document.

    This requires more complex updates to the compatibility matrix
    and release window sections.
    """
    if not file_path.exists():
        print(f"  Warning: File not found: {file_path}")
        return False

    with open(file_path) as f:
        content = f.read()

    # This is a complex file that may need manual review
    # For now, just check if it contains the version
    if version in content:
        print(f"  Version {version} already in versioning_policy.md")
        return False

    print("  Note: Manual update may be needed for versioning_policy.md")
    print(f"  - Add {version} to Release compatibility matrix")
    print("  - Update Release window section")
    print("  - Update Branch states section")

    return False


def main():
    parser = argparse.ArgumentParser(description="Update version references across documentation")
    parser.add_argument("--version", required=True, help="New vLLM Ascend version")
    parser.add_argument("--vllm-version", required=True, help="Compatible vLLM version")
    parser.add_argument("--feedback-issue", default="", help="Feedback issue URL")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without updating")

    args = parser.parse_args()

    # Normalize version (remove 'v' prefix for some contexts)
    version = args.version
    version_num = version.lstrip("v")

    variables = {
        "version": version,
        "version_num": version_num,
        "vllm_version": args.vllm_version,
        "feedback_issue": args.feedback_issue,
        "date": datetime.now().strftime("%Y-%m-%d"),
    }

    repo_root = Path(args.repo_root)
    updated_files = []

    print(f"Updating version references to {version}...")
    print(f"Compatible vLLM version: {args.vllm_version}")
    print()

    for file_config in VERSION_FILES:
        file_path = repo_root / file_config["path"]
        print(f"Processing {file_config['path']}...")

        if update_file(file_path, file_config["updates"], variables, args.dry_run):
            updated_files.append(file_config["path"])

    # Handle versioning_policy.md separately
    versioning_policy_path = repo_root / "docs/source/community/versioning_policy.md"
    print("Processing versioning_policy.md...")
    update_versioning_policy(versioning_policy_path, version, args.vllm_version, args.dry_run)

    print()
    print("Summary:")
    print(f"  Files updated: {len(updated_files)}")
    for f in updated_files:
        print(f"    - {f}")

    print()
    print("Manual updates may be needed for:")
    print("  - docs/source/community/versioning_policy.md (compatibility matrix)")
    print("  - docs/source/community/contributors.md (new contributors)")
    print("  - docs/source/user_guide/release_notes.md (release notes)")

    return 0


if __name__ == "__main__":
    exit(main())
