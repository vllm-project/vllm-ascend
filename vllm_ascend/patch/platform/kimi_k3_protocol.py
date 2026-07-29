# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 XTML marker constants used by the parser glue and tests."""

OPEN_TOKEN = "<|open|>"
CLOSE_TOKEN = "<|close|>"
SEP_TOKEN = "<|sep|>"
END_OF_MSG_TOKEN = "<|end_of_msg|>"

THINK_START = f"{OPEN_TOKEN}think{SEP_TOKEN}"
THINK_END = f"{CLOSE_TOKEN}think{SEP_TOKEN}"
RESPONSE_START = f"{OPEN_TOKEN}response{SEP_TOKEN}"
RESPONSE_END = f"{CLOSE_TOKEN}response{SEP_TOKEN}"
TOOLS_START = f"{OPEN_TOKEN}tools{SEP_TOKEN}"
TOOLS_END = f"{CLOSE_TOKEN}tools{SEP_TOKEN}"
CALL_END = f"{CLOSE_TOKEN}call{SEP_TOKEN}"
ARGUMENT_END = f"{CLOSE_TOKEN}argument{SEP_TOKEN}"
JSON_END = f"{CLOSE_TOKEN}json{SEP_TOKEN}"
MESSAGE_END = f"{CLOSE_TOKEN}message{SEP_TOKEN}"
