# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resumable recurrent-state checkpoints for hybrid attention/Mamba models.

Derived from vLLM PR #66881 (``vllm/v1/core/hybrid_checkpoint/``), Copyright
contributors to the vLLM project, Apache-2.0. The Ascend port keeps the
algorithm and re-hosts it as a plugin (see ``compat.py`` for why the core
directory cannot be edited from this repository).

See ``docs/source/user_guide/feature_guide/flashstate.md``.
"""
