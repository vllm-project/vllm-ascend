# Contributing to vLLM Ascend

You may find information about contributing to vLLM Ascend on [Developer Guide - Contributing](https://vllm-ascend.readthedocs.io/en/latest/developer_guide/contribution/index.html), including step-by-step guide to help you setup development environment, contribute first PR and test locally.

Pull request titles must use one or more repository categories, such as
`[Core]`, `[Kernel]`, `[BugFix]`, `[Doc]`, or `[Test]`. Commits must include a
Developer Certificate of Origin sign-off; use `git commit -s`.

For user-visible changes, add a release-note fragment at
`release_notes/fragments/<PR number>.yaml` in the same pull request. The PR
description must explain the problem and user impact, and include the exact
validation commands and their results. Use the repository pull-request
template to cover testing and edge-case review.
