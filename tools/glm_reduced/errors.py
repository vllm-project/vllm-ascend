# SPDX-License-Identifier: Apache-2.0
"""Error types for the GLM checkpoint reduction tool."""


class ReductionError(Exception):
    """Base class for all reduction failures. Messages must be actionable."""


class UnsupportedFormatError(ReductionError):
    """The checkpoint uses a weight format/layout this tool cannot safely reduce."""


class UnsupportedQuantError(ReductionError):
    """The quantization scheme cannot be reduced without corrupting metadata."""


class UnknownTensorError(ReductionError):
    """A tensor name did not match any known global/layer/vision pattern."""


class SafetyError(ReductionError):
    """Unsafe input/output path relationship or overwrite attempt."""


class ManifestError(ReductionError):
    """Manifest is missing, malformed or disagrees with the output files."""


class ProfileError(ReductionError):
    """Profile is unknown, or the checkpoint config violates profile rules."""
