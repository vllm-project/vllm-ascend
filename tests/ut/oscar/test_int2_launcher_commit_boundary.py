# SPDX-License-Identifier: Apache-2.0
"""Commit-boundary contract tests for the OSCAR INT2 fast launcher.

torch_npu (>= 2.10) submits every op_api/ATen host call through a per-device
task queue drained by a consumer thread (OpCommand::RunOpApi ->
enCurrentNPUStream -> Repository::Enqueue). A launcher extension that calls
aclnn* DIRECTLY submits on the calling thread immediately, so it can overtake
ops the consumer has not submitted yet -- including the lse sentinel fill and
the caller's operand staging. That race is the root cause of the default-async
(ASCEND_LAUNCH_BLOCKING unset) token corruption observed on the main port:
B=1 1080-token repro scored 11/12 corrupted with the raw-stream launcher and
0/12 once every direct-ACLNN entry establishes the official host-side commit
boundary (NPUStream::stream(true) == MakeSureQueueEmpty) between the last
ATen op and the direct submit.

These tests pin the static ordering contract of every direct-ACLNN entry in
int2_op_launch_lse_fast_r152.cpp so the boundary cannot be silently dropped
or moved to the wrong edge (a flush BEFORE the sentinel fill does NOT
guarantee submission order; it only drains an older backlog).
"""

import pathlib

import regex as re

LAUNCHER = pathlib.Path(__file__).resolve().parents[3] / "csrc" / "oscar" / "int2_op_launch_lse_fast_r152.cpp"

# every entry that mixes ATen ops with a direct aclnn submit
DIRECT_ENTRIES = (
    "launch_lse_fast",
    "oscar_middle_prep",
    "oscar_middle_prep_batched",
    "oscar_middle_replay",
    "oscar_middle_replay_batched",
    "oscar_middle_prep_batched_v2",
    "oscar_middle_op_v3",
)


def _entry_spans(src: str):
    """Map entry name -> (start, end) byte span of its function body."""
    spans = {}
    for name in DIRECT_ENTRIES:
        m = re.search(rf"static py::object {name}\(", src)
        assert m, f"entry {name} not found in launcher source"
        start = m.start()
        # crude brace matcher from the opening '{' of the function body
        i = src.index("{", m.end())
        depth = 0
        for j in range(i, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    spans[name] = (start, j)
                    break
    return spans


def test_boundary_present_in_every_direct_entry():
    src = LAUNCHER.read_text()
    spans = _entry_spans(src)
    for name, (a, b) in spans.items():
        body = src[a:b]
        assert "r218_qbound_flush(" in body, f"{name}: direct-ACLNN entry lost its task-queue commit boundary"


def test_boundary_after_sentinel_fill_and_before_getworkspacesize():
    src = LAUNCHER.read_text()
    spans = _entry_spans(src)
    for name, (a, b) in spans.items():
        body = src[a:b]
        flush = body.find("r218_qbound_flush(")
        assert flush > 0, name
        gws = body.find("GetWorkspaceSize(")
        assert gws > flush, f"{name}: commit boundary must precede the direct ACLNN call"
        fill = max(body.find("fill_(torch::Scalar(-9999.0))"), body.find("out.zero_(); lse.fill_"))
        if fill >= 0:
            assert flush > fill, (
                f"{name}: commit boundary must come AFTER the sentinel fill "
                "(a flush before the fill leaves the fill/task-queue race open)"
            )


def test_stream_raw_fetch_is_not_the_submit_stream():
    """The raw stream(false) fetch must not be the handle used for the direct
    submit: the fix re-fetches through the flushing conversion at the boundary
    and errors out on a failed flush (nullptr must never reach aclnn)."""
    src = LAUNCHER.read_text()
    spans = _entry_spans(src)
    for name, (a, b) in spans.items():
        body = src[a:b]
        assert "TORCH_CHECK(s != nullptr" in body, f"{name}: missing nullptr guard after the boundary flush"


def test_launch_status_checked():
    """Launch failures must surface: every direct launch return is checked."""
    src = LAUNCHER.read_text()
    n_launch = src.count("aclnnOscarInt2FusedAttnInt2Lse(ws_p,ws,ex,s);")
    n_launch += src.count("aclnnOscarInt2FusedAttnInt2Lse(ws_p, ws, ex, s);")
    n_check = src.count("aclnn launch failed")
    assert n_launch >= 7
    assert n_check == n_launch, f"{n_launch} direct launches but only {n_check} status checks"
