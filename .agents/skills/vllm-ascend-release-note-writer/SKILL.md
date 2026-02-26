---
name: vLLM Ascend Release Note Writer
description: You are a release note writer for vLLM Ascend project (vllm-project/vllm-ascend). You are responsible for writing release notes for vLLM Ascend.
---

# vLLM Ascend release Note Writer Skill

## Overview

You should use the `ref-past-release-notes-highlight.md` as style and category reference. Always read these first.

## When to use this skill

When a new version of vLLM Ascend is released, you should use this skill to write the release notes.

## How to use it

0. all output files should be saved under `vllm-ascend-release-note/output/$version` folder

1. Use the `fetch_commits-optimize.py` script to fetch the commits between the previous and current version.

```bash
uv run python fetch_commits-optimize.py --base-tag $LAST_TAG --head-tag $NEW_TAG --output 0-current-raw-commits.md
```

`0-current-raw-commits.md` is your raw data input.

2. Use the `commit-analysis-draft.csv` tool to analyze the commits and put them into the correct section.
`1-commit-analysis-draft.csv` is your workspace for commit by commit analysis for which commit goes into which section, whether can be ignored, and why. You can create auxilariy files in `tmp` folder.
    * You should check each commit. They are put into rows in the CSV file.
    * The CSV should have headers `title`, `pr number`, `user facing impact/summary`, `category`, `decision`, `reason`. Please brainstorm other fields as you see fit.

3. Draft the highlights note, and save it to `2-highlights-note-draft.md`.
4. Edit the draft highlights note in `2-highlights-note-draft.md`, and save it to `3-highlights-note-edit.md`. You should double and triple check with the raw commits + analysis. You can leave any uncertainty and doubts in the file, and we will discuss them together.
5. Use the format `This is the $NUMBER release candidate of $VERSION for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest) to get started.`.

## Writing style

1. To keep simple, you should only save one level of headings, starting with ###, which may include the following categories follow below order:

### Highlights

### Features

### Hardware and Operator Support

### Performance

### Dependencies

### Deprecation & Breaking Changes

### Documentation

### Others

2. Additional Inclusion Criteria

* User experience improvements (CLI enhancements, better error messages, configuration flexibility)
* Core feature (PD Disaggregation, KVCaceh, Graph mode, CP/SP, quantization)
* Breaking changes and deprecations (always include with clear impact description)
* Significant infrastructure changes (elastic scaling, distributed serving, hardware support)
* Major dependency updates (CANN/torch_npu/triton-ascend/MoonCake/Ray/transformers versions, critical library updates)
* Binary/deployment improvements (size reductions, Docker enhancements)
* Default behavior changes (default models, configuration changes that affect all users)
* Hardware compatibility expansions (310P, A2, A3, A5 support)
In the end we don't want to miss any important changes. But also don't want to spam the notes with unnecessary details.

3. Section Organization Guidelines

* **Model Support first**: Most immediately visible to users, should lead the highlights
* **Group by user impact**: Hardware/performance should focus on what users experience, not internal optimizations
* **Provide usage context**: Include relevant flags, configuration options, and practical usage information
* **Technical detail level**: Explain what features enable rather than just listing technical changes

4. Writing Tips

* Look up the PR if you are not sure about the details. The PR number at the end (#12345) can be looked up via vllm-project/vllm#12345. To get the description, you just need to call <https://api.github.com/repos/vllm-project/vllm/pulls/12345> and look at the body field.
* When writing the highlights, don't be too verbose. Focus exclusively on what users should know.
