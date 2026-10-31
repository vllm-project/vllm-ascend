"""Deduplicate wire Keys during Lookup and Load preparation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class KeyIndex:
    """One operation's canonical hashes and wire identities; discarded after preparation."""

    keys: list[str] = field(default_factory=list)
    indices: dict[str, int] = field(default_factory=dict)
    _hashes: dict[bytes, str] = field(default_factory=dict)
    _families: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)
    _parts: tuple[str, str] | None = None
    _family: dict[str, int] = field(default_factory=dict)

    def identify(self, parts: tuple[str, str], block_hash: bytes | str) -> int:
        if isinstance(block_hash, bytes):
            value = self._hashes.get(block_hash)
            if value is None:
                value = block_hash.hex()
                self._hashes[block_hash] = value
        else:
            value = block_hash
        if parts != self._parts:
            family = self._families.get(parts)
            if family is None:
                # One naming family already identifies canonical hashes uniquely. Only multiple
                # families need a wire-Key index to recognize aliases across naming boundaries.
                if len(self._families) == 1:
                    self.indices.update((key, index) for index, key in enumerate(self.keys))
                family = self._families[parts] = {}
            self._parts, self._family = parts, family
        index = self._family.get(value)
        if index is None:
            prefix, suffix = parts
            key = prefix + value + suffix
            if len(self._families) == 1:
                index = len(self.keys)
                self.keys.append(key)
            else:
                index = self.indices.get(key)
                if index is None:
                    index = len(self.keys)
                    self.indices[key] = index
                    self.keys.append(key)
            self._family[value] = index
        return index
