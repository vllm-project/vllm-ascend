# Sequence Parallelism

## Overview

Sequence Parallelism (SP) shards the token dimension across tensor-parallel
ranks around the communication boundaries of transformer layers. vLLM owns
the enablement decision through its parallel configuration.

## How to use

SP is automatically enabled for eligible MoE deployments when data parallelism
and tensor parallelism are both greater than one and a supported all-to-all
backend is selected. No Ascend-specific switch is required.

The former FlashComm environment variable and additional-config field are
deprecated and ignored. Remove them from deployment configurations; the
compatibility layer only emits a warning when it sees either setting.
