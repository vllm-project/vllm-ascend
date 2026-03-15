# Release Note Specification

This page defines the recommended format for vLLM Ascend release notes.
The goal is to make each release easier to scan for model support, hardware readiness, feature availability, dependency changes, and upgrade risks.

## Principles

- Keep the release note structured by topic instead of posting one long unordered change list.
- Put user-visible impact first, then supporting details.
- Call out model support, hardware support, and dependency requirements explicitly when they change.
- Separate new features from bug fixes and from breaking changes.
- Prefer linking every entry to the merged pull request.

## Recommended Sections

### 1. Version Header

Start with the release version and release date.

Example:

```md
## v0.16.0rc1 - 2026.03.09
```

### 2. Highlights

Use this section for the most important changes in the release.
It should be short and readable on its own.

Recommended content:

- major new model support
- major feature availability
- major performance breakthroughs
- changes that most users should notice first

### 3. Features

Describe new user-facing features and meaningful functional improvements.

Recommended content:

- new deployment capabilities
- new quantization support
- new parallelism or serving modes
- new tooling or observability features

### 4. Hardware and Operator Support

Use this section for changes tied to hardware enablement, operator availability, or backend-specific support.

Recommended content:

- new hardware platform support
- operator additions required for model execution
- hardware-specific limitations removed in this release

### 5. Performance

Use this section for measurable optimizations.

Recommended content:

- throughput or latency improvements
- memory usage improvements
- scaling improvements for multi-node or large-scale serving

When possible, include the affected model or scenario.

### 6. Dependencies

Use this section whenever the release expects users to upgrade or verify external dependencies.

Recommended content:

- CANN version changes
- PyTorch or torch_npu version changes
- Triton Ascend or other toolchain requirements
- image version assumptions when they matter for the release

### 7. Deprecation and Breaking Changes

Use this section for changes that may require action from users.

Recommended content:

- renamed configuration options
- removed or deprecated features
- behavior changes that can break existing deployments
- new required flags or migration notes

### 8. Documentation

Use this section for substantial new guides or important documentation clarifications.

Recommended content:

- new tutorials
- new deployment guides
- important troubleshooting notes

### 9. Others / Bug Fixes

Use this section for important fixes that do not fit cleanly into the sections above.

Recommended content:

- critical bug fixes
- correctness fixes
- stability fixes for specific models or deployment modes

If the list is long, summarize with one lead sentence and group related fixes together.

### 10. Known Issues

Use this section when a release still has important unresolved limitations.

Recommended content:

- unsupported combinations that are easy for users to hit
- temporary workarounds
- scenarios under active investigation

### 11. New Contributors

Use this section for contributor acknowledgements when the release process includes them.

Recommended content:

- GitHub handle
- commit id or contribution reference when helpful

## Writing Guidance

- Prefer one-sentence bullets with a clear subject and impact.
- Name the model, hardware, feature, or subsystem explicitly.
- Avoid vague bullets such as "optimize performance" without scope.
- If a change is experimental, label it clearly.
- If a change only affects a specific platform such as 310P or A3, say so directly.
- If users need to take action after upgrading, place that note in `Dependencies` or `Deprecation and Breaking Changes` instead of hiding it in another section.

## Recommended Template

```md
## vX.Y.Z - YYYY.MM.DD

Short release introduction.

### Highlights

- Most important user-visible change. [#PR]

### Features

- New feature or capability. [#PR]

### Hardware and Operator Support

- New hardware or operator support. [#PR]

### Performance

- Performance improvement with affected scope. [#PR]

### Dependencies

- Dependency upgrade or requirement. [#PR]

### Deprecation and Breaking Changes

- Breaking change or migration note. [#PR]

### Documentation

- New guide or major clarification. [#PR]

### Others

- Important fix or cleanup. [#PR]

### Known Issues

- Remaining limitation and workaround.

### New Contributors

- @contributor
```

## Relationship to Current Release Notes

The current release notes in this repository already use most of the sections above.
When preparing a new release, use this specification as the checklist for completeness and consistency.
