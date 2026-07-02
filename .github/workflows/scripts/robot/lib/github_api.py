"""GitHub Issues API helpers for robot workflows.

All functions read ``GITHUB_TOKEN`` and ``REPO`` from the environment.
"""

import os

import requests

DEFAULT_TIMEOUT = 30


def _api_base() -> str:
    return f"https://api.github.com/repos/{os.environ['REPO']}"


def _api_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }


def post_comment(issue_number: str | int, body: str) -> None:
    """Post a comment on the given issue or PR.

    Args:
        issue_number: Issue or PR number.
        body: The Markdown comment body.

    Raises:
        requests.HTTPError: If the API call fails.
    """
    url = f"{_api_base()}/issues/{issue_number}/comments"
    resp = requests.post(url, headers=_api_headers(), json={"body": body}, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    print(f"Comment posted to #{issue_number}")


def add_labels(issue_number: str | int, labels: list[str]) -> None:
    """Add labels to an issue or PR.

    Args:
        issue_number: Issue or PR number.
        labels: List of label names to add.

    Raises:
        requests.HTTPError: If the API call fails.
    """
    if not labels:
        return
    url = f"{_api_base()}/issues/{issue_number}/labels"
    resp = requests.post(url, headers=_api_headers(), json={"labels": labels}, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    print(f"Added labels {labels} to #{issue_number}")


def remove_label(issue_number: str | int, label: str) -> None:
    """Remove a label from an issue or PR.  404 is silently ignored.

    Args:
        issue_number: Issue or PR number.
        label: Label name to remove.

    Raises:
        requests.HTTPError: If the API call fails for a reason other than 404.
    """
    url = f"{_api_base()}/issues/{issue_number}/labels/{label}"
    resp = requests.delete(url, headers=_api_headers(), timeout=DEFAULT_TIMEOUT)
    if resp.status_code == 404:
        print(f"Label '{label}' not present on #{issue_number}, skipping remove")
        return
    resp.raise_for_status()
    print(f"Removed label '{label}' from #{issue_number}")


def get_labels(issue_number: str | int) -> list[str]:
    """Return the label names currently on an issue or PR.

    Args:
        issue_number: Issue or PR number.

    Returns:
        List of label name strings.

    Raises:
        requests.HTTPError: If the API call fails.
    """
    url = f"{_api_base()}/issues/{issue_number}"
    resp = requests.get(url, headers=_api_headers(), timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return [lb["name"] for lb in resp.json().get("labels", [])]


def manage_labels(
    issue_number: str | int,
    add: list[str] | None = None,
    remove: list[str] | None = None,
) -> None:
    """Atomically remove labels then add labels on an issue or PR.

    Args:
        issue_number: Issue or PR number.
        add: Labels to add.
        remove: Labels to remove (404s are silently ignored).

    Raises:
        requests.HTTPError: If any non-404 API call fails.
    """
    for label in remove or []:
        try:
            remove_label(issue_number, label)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"Label '{label}' not present on #{issue_number}, skipping remove")
            else:
                raise
    add_labels(issue_number, add or [])
