#!/usr/bin/env bash
#
# Resolve a pull request number into the git ref the image builds check out.
#
# This is the single place that turns a user-facing "pr_number" dispatch input
# into the internal "source_ref" consumed by the reusable image build
# workflows, so the manual entries do not each reimplement the GitHub API
# lookup.
#
# Inputs (environment variables):
#   PR_NUMBER    Pull request number. Must be a positive integer.
#   PR_REF_MODE  'merge' (default) builds refs/pull/<n>/merge, i.e. the PR
#                already merged into its base branch. 'head' builds only the
#                PR head commit (what /presmoke uses today).
#   GH_REPO      'owner/repo' of the base repository.
#   GH_TOKEN     Token used for the GitHub API. Optional for public repos but
#                recommended to avoid rate limits.
#   GITHUB_OUTPUT File the step outputs are appended to (set by Actions).
#
# Outputs:
#   source_ref  Git ref to build, e.g. refs/pull/123/merge.
#   base_ref    Branch the PR targets, used for the published tag.
#   head_sha    Current PR head commit, for traceability.
#   pr_title    PR title (single line), for traceability.
set -euo pipefail

PR_NUMBER="${PR_NUMBER:-}"
PR_REF_MODE="${PR_REF_MODE:-merge}"
GH_REPO="${GH_REPO:?GH_REPO must be set to owner/repo}"
GH_TOKEN="${GH_TOKEN:-}"
GITHUB_OUTPUT="${GITHUB_OUTPUT:-/dev/stdout}"

if ! [[ "$PR_NUMBER" =~ ^[0-9]+$ ]] || [ "$PR_NUMBER" = "0" ]; then
  echo "::error::pr_number must be a positive integer, got '${PR_NUMBER}'"
  exit 1
fi

case "$PR_REF_MODE" in
  merge|head) ;;
  *)
    echo "::error::pr_ref_mode must be 'merge' or 'head', got '${PR_REF_MODE}'"
    exit 1
    ;;
esac

api() {
  local path="$1"
  if [ -n "$GH_TOKEN" ]; then
    curl -fsSL \
      -H "Authorization: Bearer ${GH_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/${GH_REPO}${path}"
  else
    curl -fsSL \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/${GH_REPO}${path}"
  fi
}

if ! PR_JSON=$(api "/pulls/${PR_NUMBER}"); then
  echo "::error::Failed to fetch PR #${PR_NUMBER} from ${GH_REPO} (does it exist and is it visible to the token?)"
  exit 1
fi

STATE=$(jq -r '.state' <<<"$PR_JSON")
BASE_REPO=$(jq -r '.base.repo.full_name' <<<"$PR_JSON")
BASE_REF=$(jq -r '.base.ref' <<<"$PR_JSON")
HEAD_SHA=$(jq -r '.head.sha' <<<"$PR_JSON")
PR_TITLE=$(jq -r '.title' <<<"$PR_JSON" | tr '\n\r' '  ')

if [ "$BASE_REPO" != "$GH_REPO" ]; then
  echo "::error::PR #${PR_NUMBER} targets ${BASE_REPO}, not ${GH_REPO}; refusing to build it here"
  exit 1
fi
if [ "$STATE" != "open" ]; then
  echo "::error::PR #${PR_NUMBER} is '${STATE}'; only open pull requests can be built"
  exit 1
fi
if [ -z "$BASE_REF" ] || [ -z "$HEAD_SHA" ]; then
  echo "::error::PR #${PR_NUMBER} response is missing base.ref or head.sha"
  exit 1
fi

if [ "$PR_REF_MODE" = "merge" ]; then
  # GitHub computes mergeable asynchronously; the first read can be null while
  # the merge commit is being generated, so poll a few times before failing.
  MERGEABLE=$(jq -r '.mergeable' <<<"$PR_JSON")
  for _ in 1 2 3 4 5; do
    if [ "$MERGEABLE" = "true" ] || [ "$MERGEABLE" = "false" ]; then
      break
    fi
    sleep 5
    if ! PR_JSON=$(api "/pulls/${PR_NUMBER}"); then
      echo "::error::Failed to re-read PR #${PR_NUMBER} while waiting for mergeability"
      exit 1
    fi
    MERGEABLE=$(jq -r '.mergeable' <<<"$PR_JSON")
  done
  if [ "$MERGEABLE" != "true" ]; then
    echo "::error::PR #${PR_NUMBER} cannot be merged into ${BASE_REF} (mergeable=${MERGEABLE:-unknown}). Resolve the conflicts, or rerun with pr_ref_mode=head to build the head commit only."
    exit 1
  fi
  SOURCE_REF="refs/pull/${PR_NUMBER}/merge"
else
  SOURCE_REF="refs/pull/${PR_NUMBER}/head"
fi

{
  echo "source_ref=${SOURCE_REF}"
  echo "base_ref=${BASE_REF}"
  echo "head_sha=${HEAD_SHA}"
  echo "pr_title=${PR_TITLE}"
} >> "$GITHUB_OUTPUT"

{
  echo "### Building PR #${PR_NUMBER}"
  echo ""
  echo "- ref: \`${SOURCE_REF}\` (mode: ${PR_REF_MODE})"
  echo "- base branch: \`${BASE_REF}\`"
  echo "- head commit: \`${HEAD_SHA}\`"
  echo "- title: ${PR_TITLE}"
} >> "${GITHUB_STEP_SUMMARY:-/dev/null}"

echo "Resolved PR #${PR_NUMBER}: ${SOURCE_REF} (base=${BASE_REF}, head=${HEAD_SHA})"
