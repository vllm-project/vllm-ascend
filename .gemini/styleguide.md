# Pull Request Summary Style Guide

## Pull Request Summary Format

The summary should follow the format:

   ```markdown
    ### What this PR does / why we need it?
    <!--
    - Please clarify what changes you are proposing. The purpose of this section is to outline the changes and how this PR fixes the issue.
    If possible, please consider writing useful notes for better and faster reviews in your PR.

    - Please clarify why the changes are needed. For instance, the use case and bug description.

    - Fixes #
    -->

    ### Does this PR introduce _any_ user-facing change?
    <!--
    Note that it means *any* user-facing change including all aspects such as API, interface or other behavior changes.
    Documentation-only updates are not considered user-facing changes.
    -->

    ### How was this patch tested?
    <!--
    CI passed with new added/existing test.
    If it was tested in a way different from regular unit tests, please clarify how you tested step by step, ideally copy and paste-able, so that other reviewers can test and check, and descendants can verify in the future.
    If tests were not added, please describe why they were not added and/or why it was difficult to add.
    -->
   ```

## Pull Request Title Format

The summary should also refresh the Pull Request Title to follow the format:

    ```txt
    [Branch][Module][Action] Pull Request Title
    ```

- Branch: The branch name where the PR is based.
- Module: The module or component being changed. It includes but is not limited to the following:
    - [Attention]
    - [Ops]
    - [Doc]
    - [Test]
    - [CI]
    - [Benchmark]
- Action: The action being performed. It includes but is not limited to the following:
    - [BugFix]
    - [Feature]
    - [Misc]
