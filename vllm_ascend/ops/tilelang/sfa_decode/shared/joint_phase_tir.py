# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tilelang import tvm
from tvm import tirx


def instruction(op, *args):
    return tirx.Evaluate(tirx.call_intrin("void", tvm.ir.Op.get("tl." + op), *args))


def engine_block(kind, stmt):
    attrs = {"vector_count": tirx.IntImm("int32", 2)} if kind == "VECTOR" else {}
    return tirx.SBlock([], [], [], kind, stmt, annotations=attrs)


def checkpoint():
    def half(name, send, receive, pipe):
        ops = [
            instruction("ascend_pipe_barrier", "PIPE_ALL"),
            instruction("ascend_cross_core_set_flag", 2, pipe, send),
            instruction("ascend_cross_core_wait_flag", 2, "PIPE_MTE2", receive),
            instruction("ascend_pipe_barrier", "PIPE_ALL"),
        ]
        return engine_block(name, tirx.SeqStmt(ops))

    return tirx.SeqStmt([half("CUBE", 15, 14, "PIPE_FIX"), half("VECTOR", 14, 15, "PIPE_MTE3")])


def body_at(fn, shared, logical_index):
    grid = []

    def strip(node):
        if (
            isinstance(node, tirx.AttrStmt)
            and node.attr_key == "thread_extent"
            and node.node.thread_tag == "blockIdx.x"
        ):
            grid.append(node.node.var)
            return node.body
        return None

    stmt = tirx.stmt_functor.ir_transform(fn.body, None, strip, ["tirx.AttrStmt"])
    assert len(grid) == 1, grid
    mapper = {p: shared[p.name] for p in fn.params}
    mapper[grid[0]] = logical_index
    return tirx.stmt_functor.substitute(stmt, mapper)


def grid_barrier(*, pair_at_entry=True):
    def rendezvous(kind, pipeline, flag):
        statements = [
            instruction("ascend_pipe_barrier", "PIPE_ALL"),
            instruction("ascend_cross_core_set_flag", 0, pipeline, flag),
            instruction("ascend_cross_core_wait_flag", 0, pipeline, flag),
        ]
        return engine_block(kind, tirx.SeqStmt(statements))

    barriers = [rendezvous("CUBE", "PIPE_FIX", 12), rendezvous("VECTOR", "PIPE_MTE3", 13), checkpoint()]
    if pair_at_entry:
        barriers.insert(0, checkpoint())
    return tirx.SeqStmt(barriers)


def rewrite_host(hostmod, old_workers, workers, old_mem, ub, *, kernel_name="main_kernel"):
    changed = []

    def rewrite(n):
        if (
            isinstance(n, tirx.Call)
            and n.args
            and isinstance(n.args[0], tirx.StringImm)
            and n.args[0].value == "main_kernel"
        ):
            args = list(n.args)
            assert int(args[-2]) == old_workers and int(args[-1]) == old_mem
            args[0] = tirx.StringImm(kernel_name)
            args[-2] = tirx.IntImm(args[-2].dtype, workers)
            args[-1] = tirx.IntImm(args[-1].dtype, ub)
            changed.append(True)
            return tirx.Call(n.dtype, n.op, args)
        return None

    result = tvm.IRModule(
        {
            v: f.with_body(tirx.stmt_functor.ir_transform(f.body, None, rewrite, ["tirx.Call"]))
            for v, f in hostmod.functions.items()
        },
        attrs=hostmod.attrs,
    )
    assert len(changed) == 1
    return result


def per_kind(stmt, kind):
    scopes = []

    def collect(x):
        if isinstance(x, tirx.SBlock) and x.name_hint in ("VECTOR", "CUBE"):
            scopes.append(x.name_hint)

    tirx.stmt_functor.post_order_visit(stmt, collect)
    if kind == "mixed":
        return stmt
    if kind in scopes:
        assert set(scopes) == {kind}, scopes
        return stmt
    return engine_block(kind, stmt)


def _cube_tile_fence(body):
    # L1 can be overwritten after MTE1, L0A/B after MAC and L0C after FIX.
    pipes = ("MTE1_MTE2", "M_MTE1", "FIX_M")
    # Flag IDs 0..7 are shared with the events assigned by the TileLang scheduler.
    flag = 7
    unavailable = False
    analyzer = tvm.arith.Analyzer()

    def inspect(node):
        nonlocal unavailable
        if isinstance(node, tirx.Call) and getattr(node.op, "name", None) in (
            "tl.ascend_set_flag",
            "tl.ascend_wait_flag",
        ):
            direction, event = node.args
            if not isinstance(direction, tirx.StringImm):
                unavailable = True
            elif direction.value in pipes:
                bounds = analyzer.const_int_bound(event)
                unavailable |= bounds.min_value <= flag <= bounds.max_value

    tirx.stmt_functor.post_order_visit(body, inspect)
    if unavailable:
        return None
    events = [
        tirx.Evaluate(tirx.call_intrin("void", tvm.ir.Op.get("tl.ascend_" + action + "_flag"), pipe, flag))
        for action in ("set", "wait")
        for pipe in pipes
    ]
    return per_kind(tirx.SeqStmt(events), "CUBE")


def emit_stage(fn, params, block, workers, work, kind, *, independent_tiles=False, first_tile=0):
    turn = tirx.Var("turn", "int32")
    factor = 2 if kind == "VECTOR" else 1
    sid = tirx.Var("vector_in_pair", "int32")
    idx = (turn * workers + block) * factor + (sid if factor == 2 else 0)
    body = body_at(fn, params, idx + first_tile if first_tile else idx)

    def predicated(statement):
        if work % (workers * factor) == 0:
            return statement
        return tirx.IfThenElse(idx < work, statement, None)

    if kind == "mixed":
        steps = predicated(tirx.SeqStmt([body, checkpoint()]))
    else:
        drain = tirx.Evaluate(tirx.call_intrin("void", tvm.ir.Op.get("tl.ascend_pipe_barrier"), "PIPE_ALL"))
        if independent_tiles and kind == "CUBE" and work > workers:
            fence = _cube_tile_fence(body)
            if fence is not None:
                drain = tirx.IfThenElse(turn + 1 < (work + workers - 1) // workers, fence, drain)
        steps = tirx.SeqStmt([predicated(body), drain])
        if factor == 2:
            cthread = tirx.IterVar(tvm.ir.Range.from_min_extent(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
            steps = tirx.AttrStmt(cthread, "thread_extent", 2, steps)
        steps = per_kind(steps, kind)
    return tirx.For(turn, 0, (work + workers * factor - 1) // (workers * factor), tirx.ForKind.SERIAL, steps)
