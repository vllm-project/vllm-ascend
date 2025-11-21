from vllm.entrypoints.openai import tool_parsers

from .pangu_tool_parser import PanguToolParser


def register_tool():
    tool_parsers.__all__.append("PanguToolParser")
