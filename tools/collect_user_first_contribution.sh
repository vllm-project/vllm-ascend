#!/usr/bin/env bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from https://github.com/vllm-project/vllm/tree/main/tools
#

set -euo pipefail

# Default configuration
DEFAULT_REPO="vllm-project/vllm-ascend"
DEFAULT_CONTRIBUTORS_FILE="docs/source/community/contributors.md"

function usage() {
  echo "This script collects contributors' first contributions and updates the contributors.md file."
  echo "Supports incremental updates by tracking the last commit hash."
  echo ""
  echo "Please set the environment variable GITHUB_TOKEN with repo read permission."
  echo "Refer to https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28"
  echo ""
  echo "Usage: $0 [options]"
  echo "       $0 --full  # Force full refresh (ignore last commit hash)"
  echo "       $0 --help"
  echo ""
  echo "Options:"
  echo "  --full             Force full refresh, recalculate all contributors"
  echo "  --repo=OWNER/REPO  Specify GitHub repository (default: ${DEFAULT_REPO})"
  echo "  --file=PATH        Specify contributors.md path (default: ${DEFAULT_CONTRIBUTORS_FILE})"
  echo ""
  echo "Examples:"
  echo "  $0                 # Incremental update from last commit"
  echo "  $0 --full          # Full refresh"
}

# Parse arguments
REPO="${DEFAULT_REPO}"
CONTRIBUTORS_FILE="${DEFAULT_CONTRIBUTORS_FILE}"
FORCE_FULL=false

for arg in "$@"; do
  case $arg in
    --help)
      usage
      exit 0
      ;;
    --full)
      FORCE_FULL=true
      shift
      ;;
    --repo=*)
      REPO="${arg#*=}"
      shift
      ;;
    --file=*)
      CONTRIBUTORS_FILE="${arg#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      usage
      exit 1
      ;;
  esac
done

GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: Please set the environment variable GITHUB_TOKEN with repo read permission."
  echo "Refer to https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28"
  exit 1
fi

# Get the script directory to find the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve contributors file path
if [[ "$CONTRIBUTORS_FILE" != /* ]]; then
  CONTRIBUTORS_FILE="${PROJECT_ROOT}/${CONTRIBUTORS_FILE}"
fi

if [ ! -f "$CONTRIBUTORS_FILE" ]; then
  echo "Error: Contributors file not found: ${CONTRIBUTORS_FILE}"
  exit 1
fi

# Change to project root for git operations
cd "$PROJECT_ROOT"

# Get current HEAD commit hash
CURRENT_HEAD=$(git rev-parse HEAD)
CURRENT_HEAD_SHORT="${CURRENT_HEAD:0:7}"

echo "Repository: ${REPO}"
echo "Contributors file: ${CONTRIBUTORS_FILE}"
echo "Current HEAD: ${CURRENT_HEAD_SHORT}"
echo ""

# Function to extract last commit hash from contributors file
get_last_commit_hash() {
  local file="$1"
  # Look for comment line with last commit hash: <!-- last_commit: abc1234 -->
  grep -o '<!-- last_commit: [a-f0-9]* -->' "$file" 2>/dev/null | sed 's/<!-- last_commit: \([a-f0-9]*\) -->/\1/' || echo ""
}

# Function to extract current contributor count from file
get_current_contributor_count() {
  local file="$1"
  # Find the first row number in the table (most recent contributor)
  grep -o '| [0-9]* |' "$file" 2>/dev/null | head -1 | grep -o '[0-9]*' || echo "0"
}

# Function to extract GitHub login from noreply email
# Format: ID+username@users.noreply.github.com or username@users.noreply.github.com
extract_login_from_noreply_email() {
  local email="$1"
  if [[ "$email" == *@users.noreply.github.com ]]; then
    # Remove the domain part
    local local_part="${email%@users.noreply.github.com}"
    # Check if it's in format "ID+username" or just "username"
    if [[ "$local_part" == *+* ]]; then
      # Format: ID+username -> extract username
      echo "${local_part#*+}"
    else
      # Format: username
      echo "$local_part"
    fi
  else
    echo ""
  fi
}

# Function to get GitHub login for a commit
get_github_login() {
  local sha="$1"
  local email="$2"
  local api_url="https://api.github.com/repos/${REPO}/commits/${sha}"
  local resp
  resp=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github.v3+json" "$api_url")
  local login
  login=$(echo "$resp" | jq -r '.author.login // empty' 2>/dev/null || echo "")

  # If no login from API, try to extract from noreply email
  if [ -z "$login" ]; then
    login=$(extract_login_from_noreply_email "$email")
  fi

  echo "$login"
}

# Check if we should do incremental update
LAST_COMMIT=""
INCREMENTAL=false

if [ "$FORCE_FULL" = false ]; then
  LAST_COMMIT=$(get_last_commit_hash "$CONTRIBUTORS_FILE")
  if [ -n "$LAST_COMMIT" ] && [ "$LAST_COMMIT" != "$CURRENT_HEAD" ]; then
    # Check if LAST_COMMIT is an ancestor of CURRENT_HEAD
    if git merge-base --is-ancestor "$LAST_COMMIT" "$CURRENT_HEAD" 2>/dev/null; then
      INCREMENTAL=true
      echo "Incremental update from commit: ${LAST_COMMIT:0:7}"
    else
      echo "Warning: Last commit ${LAST_COMMIT:0:7} is not an ancestor of current HEAD, doing full refresh."
    fi
  elif [ "$LAST_COMMIT" = "$CURRENT_HEAD" ]; then
    echo "Already up to date (HEAD matches last recorded commit)."
    echo "Use --full to force a full refresh."
    exit 0
  fi
fi

if [ "$INCREMENTAL" = true ]; then
  # Incremental update: get new commits since last commit
  echo ""
  echo "Fetching new commits..."

  # Get all commits in time order, format: sha|email|name|date
  ALLCOMMITS=$(mktemp)
  git log --pretty=format:'%H|%aE|%aN|%cI' --reverse "${LAST_COMMIT}..${CURRENT_HEAD}" > "$ALLCOMMITS"

  # Get the first commit for each author email (from all history, but we'll filter to new ones)
  ALL_HISTORY=$(mktemp)
  git log --pretty=format:'%H|%aE|%aN|%cI' --reverse --all > "$ALL_HISTORY"

  # First commit by email (from all history)
  FIRST_BY_EMAIL=$(mktemp)
  awk -F'|' '!seen[$2]++ {print $2 "|" $1 "|" $4 "|" $3}' "$ALL_HISTORY" > "$FIRST_BY_EMAIL"

  # New SHAs in this range
  NEW_SHAS=$(mktemp)
  git rev-list "${LAST_COMMIT}..${CURRENT_HEAD}" > "$NEW_SHAS"

  # Extract existing contributor logins from the file for deduplication
  EXISTING_LOGINS=$(mktemp)
  grep -oE '\[@[^]]+\]' "$CONTRIBUTORS_FILE" 2>/dev/null | sed 's/\[@//;s/\]//' | sort -u > "$EXISTING_LOGINS" || true

  # Collect new contributors (first commit is in the new range)
  NEW_CONTRIBUTORS=$(mktemp)
  count=0
  skipped=0

  while IFS='|' read -r email sha date name; do
    if grep -Fxq "$sha" "$NEW_SHAS"; then
      # Query GitHub API
      login=$(get_github_login "$sha" "$email")

      # Skip if no GitHub login
      if [ -z "$login" ]; then
        continue
      fi

      # Check if contributor already exists (deduplication)
      if grep -Fxq "$login" "$EXISTING_LOGINS"; then
        echo "Skipping duplicate contributor: $login"
        ((skipped++)) || true
        continue
      fi

      # Format date
      formatted_date=$(echo "$date" | cut -d'T' -f1 | tr '-' '/')
      short_sha="${sha:0:7}"

      echo "${login}|${sha}|${short_sha}|${formatted_date}" >> "$NEW_CONTRIBUTORS"
      ((count++)) || true
    fi
  done < "$FIRST_BY_EMAIL"

  NEW_COUNT=$(wc -l < "$NEW_CONTRIBUTORS" | tr -d ' ')
  echo "Found ${NEW_COUNT} new contributors"
  if [ "$skipped" -gt 0 ]; then
    echo "Skipped ${skipped} duplicate contributors"
  fi

  if [ "$NEW_COUNT" -eq 0 ]; then
    echo "No new contributors found."
    rm -f "$ALLCOMMITS" "$ALL_HISTORY" "$FIRST_BY_EMAIL" "$NEW_SHAS" "$NEW_CONTRIBUTORS"
    exit 0
  fi

  # Get current contributor count
  CURRENT_COUNT=$(get_current_contributor_count "$CONTRIBUTORS_FILE")
  echo "Current contributor count: ${CURRENT_COUNT}"

  # Generate new rows (sorted by date descending)
  NEW_ROWS=$(mktemp)
  sort -t'|' -k4 -r "$NEW_CONTRIBUTORS" | awk -F'|' -v start="$CURRENT_COUNT" -v repo="$REPO" '
  BEGIN { nr = start }
  {
    login = $1
    sha = $2
    short_sha = $3
    date = $4

    # All contributors now have GitHub login
    printf "| %d | [@%s](https://github.com/%s) | %s | [%s](https://github.com/%s/commit/%s) |\n", nr + 1, login, login, date, short_sha, repo, sha
    nr++
  }' > "$NEW_ROWS"

  # Update the file
  TEMP_FILE=$(mktemp)
  CURRENT_DATE=$(date +%Y-%m-%d)
  NEW_TOTAL=$((CURRENT_COUNT + NEW_COUNT))

  # Track if we just wrote the table header (to insert new rows after separator)
  WROTE_HEADER=false

  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == "<!-- last_commit:"* ]]; then
      # Skip old last_commit line
      continue
    elif [[ "$line" == "Updated on "* ]]; then
      # Skip old update date line
      continue
    elif [[ "$line" == "Every release of vLLM Ascend"* ]]; then
      # Skip old description line
      continue
    elif [[ "$line" == "| Number | Contributor | Date | Commit ID |" ]]; then
      # Insert new content before the table header
      echo "<!-- last_commit: ${CURRENT_HEAD} -->" >> "$TEMP_FILE"
      echo "" >> "$TEMP_FILE"
      echo "Every release of vLLM Ascend would not have been possible without the following contributors:" >> "$TEMP_FILE"
      echo "" >> "$TEMP_FILE"
      echo "Updated on ${CURRENT_DATE} (from commit ${LAST_COMMIT:0:7} to ${CURRENT_HEAD_SHORT}):" >> "$TEMP_FILE"
      echo "" >> "$TEMP_FILE"
      echo "$line" >> "$TEMP_FILE"
      WROTE_HEADER=true
    elif [[ "$WROTE_HEADER" == true && "$line" == "|:"* ]]; then
      # This is the separator line after header - write it, then insert new rows
      echo "$line" >> "$TEMP_FILE"
      cat "$NEW_ROWS" >> "$TEMP_FILE"
      WROTE_HEADER=false
    else
      # Update row numbers in existing rows (increment by NEW_COUNT)
      if [[ "$line" == "| "* ]]; then
        # Extract old number and increment
        # Note: Use || true to prevent pipefail from causing script exit on non-matching lines
        old_num=$(echo "$line" | grep -o '| [0-9]* |' | head -1 | grep -o '[0-9]*' || true)
        if [ -n "$old_num" ]; then
          new_num=$((old_num + NEW_COUNT))
          echo "$line" | sed "s/| ${old_num} |/| ${new_num} |/" >> "$TEMP_FILE"
        else
          echo "$line" >> "$TEMP_FILE"
        fi
      else
        echo "$line" >> "$TEMP_FILE"
      fi
    fi
  done < "$CONTRIBUTORS_FILE"

  mv "$TEMP_FILE" "$CONTRIBUTORS_FILE"

  echo ""
  echo "Done! Added ${NEW_COUNT} new contributors. Total: ${NEW_TOTAL}"

  # Cleanup
  rm -f "$ALLCOMMITS" "$ALL_HISTORY" "$FIRST_BY_EMAIL" "$NEW_SHAS" "$NEW_CONTRIBUTORS" "$NEW_ROWS" "$EXISTING_LOGINS"

else
  # Full refresh
  echo "Performing full refresh..."
  echo ""

  # All commits in time order
  ALLCOMMITS=$(mktemp)
  git log --pretty=format:'%H|%aE|%aN|%cI' --reverse --all > "$ALLCOMMITS"

  # First commit by email
  FIRST_BY_EMAIL=$(mktemp)
  awk -F'|' '!seen[$2]++ {print $2 "|" $1 "|" $4 "|" $3}' "$ALLCOMMITS" > "$FIRST_BY_EMAIL"

  # Collect all contributors
  CONTRIBUTORS_DATA=$(mktemp)
  TOTAL=$(wc -l < "$FIRST_BY_EMAIL" | tr -d ' ')
  CURRENT=0

  echo "Processing ${TOTAL} contributors..."

  while IFS='|' read -r email sha date name; do
    CURRENT=$((CURRENT + 1))
    printf "\rProcessing: %d/%d" "$CURRENT" "$TOTAL"

    login=$(get_github_login "$sha" "$email")
    formatted_date=$(echo "$date" | cut -d'T' -f1 | tr '-' '/')
    short_sha="${sha:0:7}"

    if [ -n "$login" ]; then
      echo "${login}|${sha}|${short_sha}|${formatted_date}" >> "$CONTRIBUTORS_DATA"
    fi
    # Skip contributors without GitHub login (cannot be linked to GitHub ID)
  done < "$FIRST_BY_EMAIL"

  echo ""
  echo ""

  # Deduplicate by GitHub login (same user may have multiple emails)
  # Keep the earliest commit (first occurrence) for each login
  DEDUPED_DATA=$(mktemp)
  awk -F'|' '!seen[$1]++' "$CONTRIBUTORS_DATA" > "$DEDUPED_DATA"
  mv "$DEDUPED_DATA" "$CONTRIBUTORS_DATA"

  CONTRIBUTOR_COUNT=$(wc -l < "$CONTRIBUTORS_DATA" | tr -d ' ')
  echo "Found ${CONTRIBUTOR_COUNT} unique contributors"

  # Generate new content
  NEW_SECTION=$(mktemp)
  CURRENT_DATE=$(date +%Y-%m-%d)

  {
    echo "<!-- last_commit: ${CURRENT_HEAD} -->"
    echo ""
    echo "Every release of vLLM Ascend would not have been possible without the following contributors:"
    echo ""
    echo "Updated on ${CURRENT_DATE}:"
    echo ""
    echo "| Number | Contributor | Date | Commit ID |"
    echo "|:------:|:-----------:|:-----:|:---------:|"

    sort -t'|' -k4 -r "$CONTRIBUTORS_DATA" | awk -F'|' -v total="$CONTRIBUTOR_COUNT" -v repo="$REPO" '
    BEGIN { nr = total }
    {
      login = $1
      sha = $2
      short_sha = $3
      date = $4

      # All contributors now have GitHub login
      printf "| %d | [@%s](https://github.com/%s) | %s | [%s](https://github.com/%s/commit/%s) |\n", nr, login, login, date, short_sha, repo, sha
      nr--
    }'
  } > "$NEW_SECTION"

  # Update the file
  TEMP_FILE=$(mktemp)
  FOUND_CONTRIBUTORS=false

  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == "## Contributors" ]]; then
      FOUND_CONTRIBUTORS=true
      echo "$line" >> "$TEMP_FILE"
      cat "$NEW_SECTION" >> "$TEMP_FILE"
      break
    else
      echo "$line" >> "$TEMP_FILE"
    fi
  done < "$CONTRIBUTORS_FILE"

  if ! $FOUND_CONTRIBUTORS; then
    echo "" >> "$TEMP_FILE"
    echo "## Contributors" >> "$TEMP_FILE"
    cat "$NEW_SECTION" >> "$TEMP_FILE"
    echo ""
    echo "Warning: '## Contributors' section not found, appended at the end."
  fi

  mv "$TEMP_FILE" "$CONTRIBUTORS_FILE"

  echo "Done! Contributors list has been updated in: ${CONTRIBUTORS_FILE}"

  # Cleanup
  rm -f "$ALLCOMMITS" "$FIRST_BY_EMAIL" "$CONTRIBUTORS_DATA" "$NEW_SECTION"
fi