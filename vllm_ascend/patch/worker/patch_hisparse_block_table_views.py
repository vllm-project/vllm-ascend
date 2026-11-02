# SPDX-License-Identifier: Apache-2.0
"""Reuse only persistent block-table prefix views in the frozen HiSparse worker.

The original gather, current lengths, sliding-window mask and dirty-row
clearing remain bytecode-equivalent. Only the returned Tensor view objects
are reused. The reviewed SFA consumers treat these views' metadata as read-only.
This detached candidate is inactive until explicitly installed before capture;
remove it when upstream BlockTables retains its own persistent prefix views.
"""

import ast
import hashlib
import inspect
import textwrap
import types
from functools import update_wrapper
from pathlib import Path

_BLOCK_TABLE_SHA256 = "3433729cad5aa5fceb312e0a8d9ffabb3f7327dbdd18966d576e64d090313020"


def current_table_views(owner, tables, rows):
    """Alias current buffer contents; never cache gathered block-ID values."""
    original = lambda: tuple(table[:rows] for table in tables)
    if (
        type(rows) is not int
        or not 0 <= rows <= owner.max_num_reqs
        or len(tables) != len(owner.input_block_tables)
        or any(a is not b for a, b in zip(tables, owner.input_block_tables))
    ):
        return original()
    state = getattr(owner, "_ascend_hisparse_prefix_views", None)
    if state is None:
        # Retain allocation owners for at most this buffer layout. Different
        # buffers use the original path, without discarding live graph owners.
        state = (tuple(tables), {})
        owner._ascend_hisparse_prefix_views = state
    elif len(state[0]) != len(tables) or any(a is not b for a, b in zip(state[0], tables)):
        return original()
    views = state[1].get(rows)
    if views is None:
        views = original()
        state[1][rows] = views
    return views


def bind(original):
    """Retain the exact pinned method, replacing its final view expression."""
    method = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    assert isinstance(method, ast.FunctionDef) and method.name == "gather_block_tables"
    assert not method.decorator_list
    result = method.body[-1]
    assert isinstance(result, ast.Return)
    assert ast.unparse(result.value) == "tuple((bt[:num_reqs_padded] for bt in out))"
    result.value = ast.Call(
        func=ast.Name(id="_current_views", ctx=ast.Load()),
        args=[ast.Name(id=x, ctx=ast.Load()) for x in ("self", "out", "num_reqs_padded")],
        keywords=[],
    )
    # The closure owns the immutable helper; original globals still resolve
    # the active Triton kernel and any platform bindings at every invocation.
    factory = ast.FunctionDef(
        name="_ascend_make_gather",
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg="_current_views")], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[method, ast.Return(value=ast.Name(id=method.name, ctx=ast.Load()))],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    code = compile(module, __file__, "exec")
    factory_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == factory.name)
    gather = types.FunctionType(factory_code, original.__globals__)(current_table_views)
    update_wrapper(gather, original)
    gather._ascend_hisparse_persistent_prefix_views = True
    return gather


def install():
    from vllm.v1.worker.gpu.block_table import BlockTables

    original = BlockTables.gather_block_tables
    if getattr(original, "_ascend_hisparse_persistent_prefix_views", False):
        return
    if (
        hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != _BLOCK_TABLE_SHA256
    ):
        raise RuntimeError("Re-review prefix views for the actual BlockTables source")
    BlockTables.gather_block_tables = bind(original)
