# SPDX-License-Identifier: Apache-2.0
"""Explicit boundaries for GPU Worker facilities unavailable on Ascend."""


class UnsupportedElasticEPExecutor:
    """No thread pool is constructed when elastic EP is unsupported."""

    def execute(self, execute_method: str, *args, **kwargs):
        raise NotImplementedError("Ascend does not support elastic expert-parallel reconfiguration.")

    def shutdown(self) -> None:
        # There is deliberately no executor resource to release.
        pass
