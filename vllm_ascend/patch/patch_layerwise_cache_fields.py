# SPDX-License-Identifier: Apache-2.0
"""Backport placement fields on the existing vLLM dataclass identities.

Spec registries and model modules may already hold references to these classes.
Extend the field table and constructor together so replace(), inherited specs
and pickle use the same declaration in worker and scheduler processes.
"""

from dataclasses import dataclass, field, fields
from functools import wraps
from inspect import Parameter, signature

from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheTensor


@dataclass
class _PlacementFields:
    reuse: bool = field(default=False, kw_only=True, compare=False)
    backing_id: str | None = field(default=None, kw_only=True)


def _add_field(cls, declaration):
    # A subsequently defined dataclass receives this field from its base
    # automatically. Already-decorated subclasses have their own field tables.
    for child in cls.__subclasses__():
        _add_field(child, declaration)
    table = vars(cls).get("__dataclass_fields__")
    name = declaration.name
    if table is None or name in table:
        return

    original = cls.__init__

    @wraps(original)
    def init(self, *args, **kwargs):
        value = kwargs.pop(name, declaration.default)
        original(self, *args, **kwargs)
        object.__setattr__(self, name, value)

    params = list(signature(original).parameters.values())
    index = next((i for i, param in enumerate(params) if param.kind == Parameter.VAR_KEYWORD), len(params))
    params.insert(
        index, Parameter(name, Parameter.KEYWORD_ONLY, default=declaration.default, annotation=declaration.type)
    )
    init.__signature__ = signature(original).replace(parameters=params)
    cls.__init__ = init
    cls.__annotations__ = {**vars(cls).get("__annotations__", {}), name: declaration.type}
    cls.__dataclass_fields__ = {**table, name: declaration}
    setattr(cls, name, declaration.default)


def _install_placement_fields():
    reuse, backing_id = fields(_PlacementFields)
    _add_field(KVCacheSpec, reuse)
    if backing_id.name not in KVCacheTensor.__dataclass_fields__:
        _add_field(KVCacheTensor, backing_id)
        original_eq = KVCacheTensor.__eq__

        def tensor_eq(self, other):
            result = original_eq(self, other)
            if result is NotImplemented:
                return NotImplemented
            return result and self.backing_id == other.backing_id

        KVCacheTensor.__eq__ = tensor_eq


_install_placement_fields()
