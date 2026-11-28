"""Dense paged FlashMLA inference on Ascend A3."""

from .flash_mla_a3 import flash_mla, flash_mla_metadata
from .prepared_decode import DecodePlan, flash_mla_decode, prepare_decode

__all__ = ["DecodePlan", "flash_mla", "flash_mla_metadata", "flash_mla_decode", "prepare_decode"]
