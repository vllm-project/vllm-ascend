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
  echo "Please set the environment variable GITHUB_TOKEN with repo read permission for update modes."
  echo "Refer to https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28"
  echo "The --sort-only mode does not require GitHub API access."
  echo ""
  echo "Usage: $0 [options]"
  echo "       $0 --full  # Force full refresh (ignore last commit hash)"
  echo "       $0 --sort-only  # Renumber the existing contributors table rows"
  echo "       $0 --link-check  # Drop invalid profile or commit links while updating"
  echo "       $0 --help"
  echo ""
  echo "Options:"
  echo "  --full             Force full refresh, recalculate all contributors"
  echo "  --sort-only        Keep existing row order and only recalculate row numbers"
  echo "  --link-check       Check generated profile and commit links, then drop invalid rows"
  echo "  --repo=OWNER/REPO  Specify GitHub repository (default: ${DEFAULT_REPO})"
  echo "  --file=PATH        Specify contributors.md path (default: ${DEFAULT_CONTRIBUTORS_FILE})"
  echo ""
  echo "Examples:"
  echo "  $0                 # Incremental update from last commit"
  echo "  $0 --full          # Full refresh"
  echo "  $0 --sort-only     # Renumber existing contributor rows after manual edits"
  echo "  $0 --link-check    # Incremental update with link validation"
}

# Parse arguments
REPO="${DEFAULT_REPO}"
CONTRIBUTORS_FILE="${DEFAULT_CONTRIBUTORS_FILE}"
FORCE_FULL=false
SORT_ONLY=false
LINK_CHECK=false

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
    --sort-only)
      SORT_ONLY=true
      shift
      ;;
    --link-check)
      LINK_CHECK=true
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

if [ "$SORT_ONLY" = true ] && [ "$FORCE_FULL" = true ]; then
  echo "Error: --sort-only cannot be used with --full."
  exit 1
fi

if [ "$SORT_ONLY" = true ] && [ "$LINK_CHECK" = true ]; then
  echo "Error: --sort-only cannot be used with --link-check."
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

# Function to keep existing contributor row order and refresh the Number column
renumber_contributors_table() {
  local file="$1"
  local row_count
  row_count=$(awk '
    BEGIN { in_table = 0; count = 0 }
    $0 == "| Number | Contributor | Date | Commit ID |" { in_table = 1; next }
    in_table && $0 ~ /^\|:/ { next }
    in_table && $0 ~ /^\| [0-9]+ \|/ { count++; next }
    in_table && $0 !~ /^\|/ { in_table = 0 }
    END { print count }
  ' "$file")

  if [ "$row_count" -eq 0 ]; then
    echo "Error: No contributor rows found in ${file}"
    exit 1
  fi

  local temp_file
  temp_file=$(mktemp)

  awk -v total="$row_count" '
    BEGIN { in_table = 0; nr = total }
    $0 == "| Number | Contributor | Date | Commit ID |" {
      in_table = 1
      print
      next
    }
    in_table && $0 ~ /^\|:/ {
      print
      next
    }
    in_table && $0 ~ /^\| [0-9]+ \|/ {
      sub(/^\| [0-9]+ \|/, "| " nr " |")
      print
      nr--
      next
    }
    in_table && $0 !~ /^\|/ {
      in_table = 0
      print
      next
    }
    { print }
  ' "$file" > "$temp_file"

  cat "$temp_file" > "$file"
  rm -f "$temp_file"
  echo "Renumbered ${row_count} contributor rows in: ${file}"
}

if [ "$SORT_ONLY" = true ]; then
  renumber_contributors_table "$CONTRIBUTORS_FILE"
  exit 0
fi

GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: Please set the environment variable GITHUB_TOKEN with repo read permission."
  echo "Refer to https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28"
  exit 1
fi

# Get current HEAD commit hash
CURRENT_HEAD=$(git rev-parse HEAD)
CURRENT_HEAD_SHORT="${CURRENT_HEAD:0:7}"

echo "Repository: ${REPO}"
echo "Contributors file: ${CONTRIBUTORS_FILE}"
echo "Current HEAD: ${CURRENT_HEAD_SHORT}"
echo "Link check: ${LINK_CHECK}"
echo ""

INVALID_LINK_ROWS=""
if [ "$LINK_CHECK" = true ]; then
  INVALID_LINK_ROWS=$(mktemp)
fi

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
  resp=$(curl -sSL --globoff -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github.v3+json" "$api_url") || resp=""
  local login
  login=$(echo "$resp" | jq -r '.author.login // empty' 2>/dev/null || echo "")

  # If no login from API, try to extract from noreply email
  if [ -z "$login" ]; then
    login=$(extract_login_from_noreply_email "$email")
  fi

  echo "$login"
}

github_api_status() {
  local api_url="$1"
  local status
  status=$(curl -sSL --globoff -o /dev/null -w "%{http_code}" \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    "$api_url") || status="000"
  echo "$status"
}

get_link_error_reason() {
  local api_url="$1"
  local kind="$2"
  local status
  status=$(github_api_status "$api_url")

  if [ "$status" = "200" ]; then
    echo ""
  else
    echo "Invalid ${kind}(${status})"
  fi
}

get_invalid_link_reason() {
  local login="$1"
  local sha="$2"
  local reason=""
  local commit_reason=""

  if [ -z "$login" ]; then
    reason="Invalid profile(no-login)"
  else
    reason=$(get_link_error_reason "https://api.github.com/users/${login}" "profile")
  fi

  commit_reason=$(get_link_error_reason "https://api.github.com/repos/${REPO}/commits/${sha}" "commit")
  if [ -n "$reason" ] && [ -n "$commit_reason" ]; then
    reason="${reason},${commit_reason}"
  elif [ -n "$commit_reason" ]; then
    reason="$commit_reason"
  fi

  echo "$reason"
}

record_invalid_contributor() {
  local login="$1"
  local sha="$2"
  local short_sha="$3"
  local formatted_date="$4"
  local reason="$5"

  printf "%s|%s|%s|%s|%s\n" "$login" "$sha" "$short_sha" "$formatted_date" "$reason" >> "$INVALID_LINK_ROWS"
}

filter_valid_contributors() {
  local input_file="$1"
  local output_file="$2"

  if [ "$LINK_CHECK" != true ]; then
    cp "$input_file" "$output_file"
    return
  fi

  local total
  local current
  local reason
  total=$(wc -l < "$input_file" | tr -d ' ')
  current=0
  : > "$output_file"

  echo "Checking ${total} generated contributor links..."
  while IFS='|' read -r login sha short_sha formatted_date; do
    current=$((current + 1))
    printf "\rChecking links: %d/%d" "$current" "$total"

    reason=$(get_invalid_link_reason "$login" "$sha")
    if [ -n "$reason" ]; then
      record_invalid_contributor "$login" "$sha" "$short_sha" "$formatted_date" "$reason"
      continue
    fi

    echo "${login}|${sha}|${short_sha}|${formatted_date}" >> "$output_file"
  done < "$input_file"
  echo ""
}

print_invalid_contributors() {
  if [ "$LINK_CHECK" != true ] || [ -z "${INVALID_LINK_ROWS:-}" ] || [ ! -s "$INVALID_LINK_ROWS" ]; then
    return
  fi

  echo ""
  echo "Removed invalid contributor links:"
  echo "| Number | Contributor | Date | Commit ID | Reason |"
  echo "|:------:|:-----------:|:-----:|:---------:|:------:|"
  awk -F'|' -v repo="$REPO" '
  {
    contributor = "-"
    if ($1 != "") {
      contributor = "[@" $1 "](https://github.com/" $1 ")"
    }

    commit = "-"
    if ($2 != "") {
      commit = "[" $3 "](https://github.com/" repo "/commit/" $2 ")"
    }

    printf "| - | %s | %s | %s | %s |\n", contributor, $4, commit, $5
  }' "$INVALID_LINK_ROWS"
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

  while IFS='|' read -r email sha date _name; do
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

  FILTERED_NEW_CONTRIBUTORS=$(mktemp)
  filter_valid_contributors "$NEW_CONTRIBUTORS" "$FILTERED_NEW_CONTRIBUTORS"
  mv "$FILTERED_NEW_CONTRIBUTORS" "$NEW_CONTRIBUTORS"

  DEDUPED_NEW_CONTRIBUTORS=$(mktemp)
  awk -F'|' '!seen[$1]++' "$NEW_CONTRIBUTORS" > "$DEDUPED_NEW_CONTRIBUTORS"
  mv "$DEDUPED_NEW_CONTRIBUTORS" "$NEW_CONTRIBUTORS"

  NEW_COUNT=$(wc -l < "$NEW_CONTRIBUTORS" | tr -d ' ')
  echo "Found ${NEW_COUNT} new contributors"
  if [ "$skipped" -gt 0 ]; then
    echo "Skipped ${skipped} duplicate contributors"
  fi

  if [ "$NEW_COUNT" -eq 0 ]; then
    echo "No new contributors found."
    print_invalid_contributors
    rm -f "$ALLCOMMITS" "$ALL_HISTORY" "$FIRST_BY_EMAIL" "$NEW_SHAS" "$NEW_CONTRIBUTORS" "$FILTERED_NEW_CONTRIBUTORS" "$DEDUPED_NEW_CONTRIBUTORS"
    if [ -n "${INVALID_LINK_ROWS:-}" ]; then
      rm -f "$INVALID_LINK_ROWS"
    fi
    exit 0
  fi

  # Get current contributor count
  CURRENT_COUNT=$(get_current_contributor_count "$CONTRIBUTORS_FILE")
  echo "Current contributor count: ${CURRENT_COUNT}"

  # Generate new rows (sorted by date descending)
  NEW_ROWS=$(mktemp)
  NEW_TOTAL=$((CURRENT_COUNT + NEW_COUNT))
  sort -t'|' -k4 -r "$NEW_CONTRIBUTORS" | awk -F'|' -v total="$NEW_TOTAL" -v repo="$REPO" '
  BEGIN { nr = total }
  {
    login = $1
    sha = $2
    short_sha = $3
    date = $4

    # All contributors now have GitHub login
    printf "| %d | [@%s](https://github.com/%s) | %s | [%s](https://github.com/%s/commit/%s) |\n", nr, login, login, date, short_sha, repo, sha
    nr--
  }' > "$NEW_ROWS"

  # Update the file
  TEMP_FILE=$(mktemp)
  CURRENT_DATE=$(date +%Y-%m-%d)

  # Track if we just wrote the table header (to insert new rows after separator)
  WROTE_HEADER=false
  AFTER_CONTRIBUTORS=false

  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == "## Contributors" ]]; then
      echo "$line" >> "$TEMP_FILE"
      AFTER_CONTRIBUTORS=true
      continue
    elif [[ "$AFTER_CONTRIBUTORS" == true && "$line" == "" ]]; then
      continue
    elif [[ "$line" == "<!-- last_commit:"* ]]; then
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
      AFTER_CONTRIBUTORS=false
      {
        echo "<!-- last_commit: ${CURRENT_HEAD} -->"
        echo ""
        echo "Every release of vLLM Ascend would not have been possible without the following contributors:"
        echo ""
        echo "Updated on ${CURRENT_DATE} (from commit ${LAST_COMMIT:0:7} to ${CURRENT_HEAD_SHORT}):"
        echo ""
        echo "$line"
      } >> "$TEMP_FILE"
      WROTE_HEADER=true
    elif [[ "$WROTE_HEADER" == true && "$line" == "|:"* ]]; then
      # This is the separator line after header - write it, then insert new rows
      echo "$line" >> "$TEMP_FILE"
      cat "$NEW_ROWS" >> "$TEMP_FILE"
      WROTE_HEADER=false
    else
      echo "$line" >> "$TEMP_FILE"
    fi
  done < "$CONTRIBUTORS_FILE"

  mv "$TEMP_FILE" "$CONTRIBUTORS_FILE"

  echo ""
  echo "Done! Added ${NEW_COUNT} new contributors. Total: ${NEW_TOTAL}"
  print_invalid_contributors

  # Cleanup
  rm -f "$ALLCOMMITS" "$ALL_HISTORY" "$FIRST_BY_EMAIL" "$NEW_SHAS" "$NEW_CONTRIBUTORS" "$FILTERED_NEW_CONTRIBUTORS" "$DEDUPED_NEW_CONTRIBUTORS" "$NEW_ROWS" "$EXISTING_LOGINS"
  if [ -n "${INVALID_LINK_ROWS:-}" ]; then
    rm -f "$INVALID_LINK_ROWS"
  fi

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

  while IFS='|' read -r email sha date _name; do
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

  FILTERED_CONTRIBUTORS_DATA=$(mktemp)
  filter_valid_contributors "$CONTRIBUTORS_DATA" "$FILTERED_CONTRIBUTORS_DATA"
  mv "$FILTERED_CONTRIBUTORS_DATA" "$CONTRIBUTORS_DATA"

  # Deduplicate by GitHub login (same user may have multiple emails)
  # Keep the earliest valid commit (first occurrence) for each login
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
    {
      echo ""
      echo "## Contributors"
      cat "$NEW_SECTION"
    } >> "$TEMP_FILE"
    echo ""
    echo "Warning: '## Contributors' section not found, appended at the end."
  fi

  mv "$TEMP_FILE" "$CONTRIBUTORS_FILE"

  echo "Done! Contributors list has been updated in: ${CONTRIBUTORS_FILE}"
  print_invalid_contributors

  # Cleanup
  rm -f "$ALLCOMMITS" "$FIRST_BY_EMAIL" "$CONTRIBUTORS_DATA" "$FILTERED_CONTRIBUTORS_DATA" "$NEW_SECTION"
  if [ -n "${INVALID_LINK_ROWS:-}" ]; then
    rm -f "$INVALID_LINK_ROWS"
  fi
fi
