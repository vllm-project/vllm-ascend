#!/bin/bash
# ============================================================
# aop_commit_age.sh - Look up last successful time in good_table.csv
#
# CSV format (good_table.csv):
#   name,yaml/path,link,status,vLLM Git information,vLLM-Ascend Git information,time
#
# Finds rows matching config_name where status=success,
# picks the most recent "time" column, and calculates age.
#
# Args: $1 = config_name
#       $2 = csv_path
#       $3 = max_age_days (default: 3)
#       $4 = soc (optional; filters new-schema rows)
#
# Writes to $GITHUB_OUTPUT:
#   commit_age_days  - days since last success
#   is_old           - true if older than max_age_days
#   last_status      - status from table
#   last_date        - date from table
# ============================================================
set -euo pipefail

CONFIG_NAME="${1:-}"
CSV_PATH="${GOOD_TABLE:-$2}"
MAX_AGE_DAYS="${3:-3}"
BISECT_SOC="${4:-}"

if [ -z "$CONFIG_NAME" ]; then
  echo "ERROR: no config name provided"
  exit 1
fi

if ! [[ "$MAX_AGE_DAYS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: max_age_days must be a non-negative integer, got '${MAX_AGE_DAYS}'"
  exit 1
fi

echo ">>> Looking up config : ${CONFIG_NAME}"
echo ">>> CSV path          : ${CSV_PATH}"

if [ ! -f "$CSV_PATH" ]; then
  echo ">>> CSV not found → skip"
  echo "is_old=true"       >> "$GITHUB_OUTPUT"
  echo "commit_age_days=99" >> "$GITHUB_OUTPUT"
  echo "last_status=unknown" >> "$GITHUB_OUTPUT"
  exit 0
fi

# Match the name and, for new-schema rows, the current SoC. Legacy seven-column
# rows have no SoC and remain eligible during migration.
ROWS=$(awk -F',' -v name="$CONFIG_NAME" -v soc="$BISECT_SOC" '
  NR > 1 && $1 == name && tolower($4) == "success" &&
  (soc == "" || NF < 9 || $7 == soc) { print }
' "$CSV_PATH")

if [ -z "$ROWS" ]; then
  echo ">>> No success row for '${CONFIG_NAME}' → skip"
  echo "is_old=true"       >> "$GITHUB_OUTPUT"
  echo "commit_age_days=99" >> "$GITHUB_OUTPUT"
  echo "last_status=unknown" >> "$GITHUB_OUTPUT"
  exit 0
fi

# Pick most recent success row
BEST_ROW=""
BEST_DATE=""
while IFS= read -r row; do
  date_str=$(echo "$row" | awk -F',' '{print $NF}' | xargs)
  [ -z "$date_str" ] && continue
  if [ -z "$BEST_DATE" ] || [[ "$date_str" > "$BEST_DATE" ]]; then
    BEST_ROW="$row"
    BEST_DATE="$date_str"
  fi
done <<< "$ROWS"

if [ -z "$BEST_ROW" ]; then
  echo ">>> No valid date in success rows → skip"
  echo "is_old=true"       >> "$GITHUB_OUTPUT"
  echo "commit_age_days=99" >> "$GITHUB_OUTPUT"
  echo "last_status=unknown" >> "$GITHUB_OUTPUT"
  exit 0
fi

LAST_STATUS=$(echo "$BEST_ROW" | awk -F',' '{print $4}' | xargs)
LAST_DATE="$BEST_DATE"

echo ">>> Matched row: ${BEST_ROW}"
echo "last_status=${LAST_STATUS}" >> "$GITHUB_OUTPUT"
echo "last_date=${LAST_DATE}"     >> "$GITHUB_OUTPUT"

LAST_TS=$(date -d "$LAST_DATE" +%s 2>/dev/null || true)
if [ -z "$LAST_TS" ]; then
  echo ">>> Could not parse date: ${LAST_DATE} → skip"
  echo "is_old=true"       >> "$GITHUB_OUTPUT"
  echo "commit_age_days=99" >> "$GITHUB_OUTPUT"
  exit 0
fi

NOW=$(date +%s)
AGE_DAYS=$(( (NOW - LAST_TS) / 86400 ))

echo "commit_age_days=${AGE_DAYS}" >> "$GITHUB_OUTPUT"

if [ "$AGE_DAYS" -gt "$MAX_AGE_DAYS" ]; then
  echo "is_old=true" >> "$GITHUB_OUTPUT"
  echo ">>> ${CONFIG_NAME} last_status=${LAST_STATUS} date=${LAST_DATE} age=${AGE_DAYS}d (> ${MAX_AGE_DAYS} days) → old"
else
  echo "is_old=false" >> "$GITHUB_OUTPUT"
  echo ">>> ${CONFIG_NAME} last_status=${LAST_STATUS} date=${LAST_DATE} age=${AGE_DAYS}d (<= ${MAX_AGE_DAYS} days) → recent"
fi
