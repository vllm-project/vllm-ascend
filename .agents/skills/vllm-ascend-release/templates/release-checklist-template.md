### Release Checklist

**Release Version**:  ${VERSION}
**Release Branch**:  ${BRANCH}
**Release Date**:  ${DATE}
**Release Manager**:  @${MANAGER}


### Prepare Release Note

- [ ] Create a new issue for release feedback ${FEEDBACK_ISSUE_URL}
- [ ] Upgrade vllm version to the new version for CI and Dockerfile
- [ ] Write the release note PR. ${RELEASE_NOTE_PR_URL}

  - [ ] Update the feedback issue link in docs/source/faqs.md

  - [ ] Add release note to docs/source/user_guide/release_notes.md

  - [ ] Update release version in README.md and README.zh.md (Getting Started and Branch section)

  - [ ] Update version info in docs/source/community/versioning_policy.md(Release compatibility matrix, Release window and Branch states section)

  - [ ] Update contributor info in docs/source/community/contributors.md

  - [ ] Update package version in docs/conf.py


### Bug need Solve

<!-- AI-GENERATED: This section will be populated by the release skill -->
<!-- The skill scans open bugs and identifies release-blocking issues -->
${BUG_LIST}

### PR need Merge

<!-- AI-GENERATED: This section will be populated by the release skill -->
<!-- The skill identifies PRs that should be merged before release -->
${PR_LIST}

### Functional Test

<!-- AI-GENERATED: This section will be populated by the release skill -->
<!-- Lists features/models that need manual testing (not covered by CI) -->
<!-- Also tracks unresolved items from previous release feedback -->

#### Manual Testing Required
${MANUAL_TEST_ITEMS}

#### Previous Feedback Status
${FEEDBACK_STATUS}

### Nightly Status

<!-- AI-GENERATED: This section will be populated by the release skill -->
<!-- Analyzes latest Nightly-A3 and Nightly-A2 CI runs -->
<!-- Uses extract_and_analyze.py to extract and categorize failures -->

| Workflow | Status | Failed Jobs | Code Bugs | Env Flakes | Run |
|----------|--------|-------------|-----------|------------|-----|
${NIGHTLY_TABLE}

#### Code Bugs Need Fix
${NIGHTLY_CODE_BUGS}

### Doc Test

- [ ] Tutorial is updated.
- [ ] User Guide is updated.
- [ ] Developer Guide is updated.


### Prepare Artifacts

- [ ] Docker image is ready.
- [ ] Wheel package is ready.


### Release Step

- [ ] Release note PR is merged.
- [ ] Post the release on GitHub release page.
- [ ] Generate official doc page on <https://app.readthedocs.org/dashboard/>
- [ ] Wait for the wheel package to be available on <https://pypi.org/project/vllm-ascend>
- [ ] Wait for the docker image to be available on <https://quay.io/ascend/vllm-ascend>
- [ ] Upload 310p wheel to Github release page
- [ ] Broadcast the release news (By message, blog , etc)
- [ ] Close this issue
