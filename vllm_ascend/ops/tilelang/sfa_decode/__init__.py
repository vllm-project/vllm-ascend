# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TileLang kernels for replicated GLM-5.3-Flash NoPE attention on Ascend 950."""


def compile_mtp_decode(requests, index_pages, main_pages, ring_pages, **options):
    """Compile the target-model graph for MTP5, respecting each page table's physical width."""
    from .mtp.entry import compile_mtp_decode as compile_graph

    if options.get("table_columns") is not None:
        options["table_columns"] = tuple(options["table_columns"])
    return compile_graph(requests, index_pages, main_pages, ring_pages, **options)
