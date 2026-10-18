"""Memcache adapter; native registration and session mechanics stay in this package."""

from .backend import MemcacheBackend

__all__ = ["MemcacheBackend"]
