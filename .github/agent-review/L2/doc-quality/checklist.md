# L2 Documentation Quality Domain Checklist

- `L2-doc-quality#upstream-rule`: Review changed documentation against the
  pinned `ascend-ci/doc-quality` standards in the adjacent `upstream/` directory.
  Start with `upstream/common/h2_registry.json`. Read each applicable dimension's
  `SKILL.md` and follow its registered rule files, scope and exemptions. In an
  English finding, identify the upstream topic and clause and explain the
  consequence at the changed line. Map upstream F to `MUST-FIX`, S to
  `SHOULD-FIX`, and N or H to `NIT`.

For Chinese `.po` files, review changed entries that have a nonempty translation
(`msgstr`). Read each multiline string in full and compare the translation with
the English source (`msgid`). Check that it preserves the meaning, placeholders
and markup. Skip the catalog header and obsolete entries.

If the author says they withdrew a documentation change, check the current PR
diff against the original finding. You can verify the withdrawal without the
upstream rules. If the change was reverted and the issue is gone, mark the
original finding `resolved` and say what was withdrawn. If the text moved and
the issue persists, keep the finding `open` and cite its new location.
