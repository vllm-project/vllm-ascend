#!/usr/bin/env python3
"""
Update version references across documentation files.

This script loads configuration from references/version-files.yaml and updates
version numbers and compatibility information in various files for a new release.

Usage:
    python update_version_references.py \
        --version v0.15.0rc1 \
        --vllm-version v0.15.0 \
        --feedback-issue https://github.com/vllm-project/vllm-ascend/issues/1234
"""

import argparse
import glob
import re
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    """Load version files configuration from YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def expand_glob_pattern(repo_root: Path, pattern: str) -> list[Path]:
    """Expand glob patterns to actual file paths."""
    if "*" in pattern:
        # Use glob to expand the pattern
        full_pattern = str(repo_root / pattern)
        matches = glob.glob(full_pattern)
        return [Path(m) for m in matches]
    else:
        # Single file path
        return [repo_root / pattern]


def build_regex_pattern(pattern_template: str) -> str:
    """
    Convert a template pattern like 'pip install vllm-ascend==X.X.X'
    to a regex pattern like 'pip install vllm-ascend==[\\d.]+(?:rc\\d+)?'
    """
    # Escape special regex characters except our placeholder
    escaped = re.escape(pattern_template)
    # Replace escaped X.X.X placeholder with version regex
    escaped = escaped.replace(r"X\.X\.X", r"[\d.]+(?:rc\d+)?")
    # Replace escaped XXXX (issue number) with digit regex
    escaped = escaped.replace("XXXX", r"\d+")
    return escaped


def update_file_content(
    file_path: Path,
    updates: list[dict],
    variables: dict,
    dry_run: bool = False,
) -> tuple[bool, list[str]]:
    """
    Update a single file with version references.

    Returns:
        Tuple of (modified: bool, messages: list of update messages)
    """
    messages = []

    if not file_path.exists():
        messages.append(f"  Warning: File not found: {file_path}")
        return False, messages

    with open(file_path) as f:
        content = f.read()

    modified = False

    for update in updates:
        # Check if this update has a regex pattern or a template pattern
        if "pattern" in update:
            pattern_template = update["pattern"]
            # Build regex from template pattern
            regex_pattern = build_regex_pattern(pattern_template)

            # Build replacement string
            if "replacement" in update:
                replacement = update["replacement"].format(**variables)
            else:
                # Use the pattern template with version substituted
                replacement = pattern_template.replace("X.X.X", variables.get("version_num", ""))
                replacement = replacement.replace("XXXX", variables.get("feedback_issue_num", ""))

            new_content = re.sub(regex_pattern, replacement, content)
            if new_content != content:
                content = new_content
                modified = True
                messages.append(f"  Updated pattern: {pattern_template[:50]}...")

        elif "variable" in update:
            # Handle variable-based updates (e.g., VLLM_VERSION in workflow files)
            var_name = update["variable"]

            # Common patterns for variable definitions
            patterns = [
                # YAML: VLLM_VERSION: "0.15.0"
                (
                    rf'({var_name}:\s*["\']?)[\d.]+(?:rc\d+)?(["\']?)',
                    rf"\g<1>{variables.get('vllm_version_num', '')}\g<2>",
                ),
                # Shell: VLLM_VERSION=0.15.0
                (rf"({var_name}=)[\d.]+(?:rc\d+)?", rf"\g<1>{variables.get('vllm_version_num', '')}"),
                # Docker ARG: ARG VLLM_VERSION=0.15.0
                (rf"(ARG\s+{var_name}=)[\d.]+(?:rc\d+)?", rf"\g<1>{variables.get('vllm_version_num', '')}"),
                # ENV: ENV VLLM_VERSION=0.15.0
                (rf"(ENV\s+{var_name}=)[\d.]+(?:rc\d+)?", rf"\g<1>{variables.get('vllm_version_num', '')}"),
            ]

            for pattern, replacement in patterns:
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    modified = True
                    messages.append(f"  Updated variable: {var_name}")
                    break

        elif "action" in update:
            # This update requires manual action
            action = update["action"]
            description = update.get("description", "")
            messages.append(f"  Manual action required: {action}")
            if description:
                messages.append(f"    - {description}")

    if modified:
        if dry_run:
            messages.append(f"  [DRY RUN] Would update {file_path}")
        else:
            with open(file_path, "w") as f:
                f.write(content)
            messages.append(f"  Updated: {file_path}")

    return modified, messages


def process_file_config(
    repo_root: Path,
    file_config: dict,
    variables: dict,
    dry_run: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """
    Process a single file configuration entry.

    Returns:
        Tuple of (updated_files, manual_files, messages)
    """
    updated_files = []
    manual_files = []
    messages = []

    path_pattern = file_config["path"]
    updates = file_config.get("updates", [])

    # Expand glob patterns
    file_paths = expand_glob_pattern(repo_root, path_pattern)

    if not file_paths:
        messages.append(f"Processing {path_pattern}...")
        messages.append(f"  Warning: No files matched pattern: {path_pattern}")
        return updated_files, manual_files, messages

    for file_path in file_paths:
        rel_path = file_path.relative_to(repo_root) if file_path.is_relative_to(repo_root) else file_path
        messages.append(f"Processing {rel_path}...")

        # Check if all updates are manual actions
        has_auto_updates = any("pattern" in u or "variable" in u for u in updates)
        has_manual_updates = any("action" in u for u in updates)

        if has_auto_updates:
            modified, update_messages = update_file_content(file_path, updates, variables, dry_run)
            messages.extend(update_messages)
            if modified:
                updated_files.append(str(rel_path))

        if has_manual_updates:
            manual_files.append(str(rel_path))
            for update in updates:
                if "action" in update:
                    messages.append(f"  Manual action: {update['action']}")
                    if "description" in update:
                        messages.append(f"    - {update['description']}")

    return updated_files, manual_files, messages


def main():
    parser = argparse.ArgumentParser(description="Update version references across documentation")
    parser.add_argument("--version", required=True, help="New vLLM Ascend version (e.g., v0.15.0rc1)")
    parser.add_argument("--vllm-version", required=True, help="Compatible vLLM version (e.g., v0.15.0)")
    parser.add_argument("--feedback-issue", default="", help="Feedback issue URL")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--config", default=None, help="Config file path (default: references/version-files.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without updating")

    args = parser.parse_args()

    # Determine config path
    script_dir = Path(__file__).parent
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = script_dir.parent / "references" / "version-files.yaml"

    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Normalize versions
    version = args.version
    version_num = version.lstrip("v")
    vllm_version = args.vllm_version
    vllm_version_num = vllm_version.lstrip("v")

    # Extract feedback issue number if URL provided
    feedback_issue_num = ""
    if args.feedback_issue:
        match = re.search(r"/issues/(\d+)", args.feedback_issue)
        if match:
            feedback_issue_num = match.group(1)

    variables = {
        "version": version,
        "version_num": version_num,
        "vllm_version": vllm_version,
        "vllm_version_num": vllm_version_num,
        "feedback_issue": args.feedback_issue,
        "feedback_issue_num": feedback_issue_num,
        "date": datetime.now().strftime("%Y-%m-%d"),
    }

    repo_root = Path(args.repo_root).resolve()

    print(f"Updating version references to {version}...")
    print(f"Compatible vLLM version: {vllm_version}")
    print(f"Config file: {config_path}")
    print(f"Repository root: {repo_root}")
    print()

    all_updated_files = []
    all_manual_files = []

    # Get update order from config, or use file order
    update_order = config.get("update_order", {})
    files_config = config.get("files", [])

    # Sort files by update order if specified
    if update_order:
        # Create a mapping from path to order
        path_to_order = {v: int(k) for k, v in update_order.items()}

        def get_order(file_config):
            path = file_config["path"]
            return path_to_order.get(path, 999)

        files_config = sorted(files_config, key=get_order)

    # Process each file configuration
    for file_config in files_config:
        updated, manual, messages = process_file_config(repo_root, file_config, variables, args.dry_run)
        all_updated_files.extend(updated)
        all_manual_files.extend(manual)
        for msg in messages:
            print(msg)
        print()

    # Print summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Files automatically updated: {len(all_updated_files)}")
    for f in all_updated_files:
        print(f"  ✓ {f}")

    if all_manual_files:
        print()
        print(f"Files requiring manual updates: {len(set(all_manual_files))}")
        for f in sorted(set(all_manual_files)):
            print(f"  ⚠ {f}")

    # Print validation hints from config
    validation = config.get("validation", [])
    if validation:
        print()
        print("Validation commands:")
        for cmd in validation:
            print(f"  $ {cmd}")

    return 0


if __name__ == "__main__":
    exit(main())
