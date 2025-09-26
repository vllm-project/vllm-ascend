from vllm import reasoning
from .pangu_reasoning_parser import PanguReasoningParser


def register_reasoning():
    reasoning.__all__.append("PanguReasoningParser")