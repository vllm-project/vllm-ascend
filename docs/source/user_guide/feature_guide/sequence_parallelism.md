# Sequence Parallelism

## Overview

Sequence Parallelism (SP) shards the token dimension across tensor-parallel
ranks around the communication boundaries of transformer layers. vLLM owns

## How to use

Automatically enabled when DP>1, TP>1 are set, and specific all2all backend with moe model.
The former FlashComm settings are removed.

```text
FlashComm is deprecated
```

Use the upstream SP configuration shown above instead.
