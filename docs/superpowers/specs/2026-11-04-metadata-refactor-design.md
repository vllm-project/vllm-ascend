# AscendStore metadata refactor

The user approved the incremental refactor discussed in chat: remove redundant
arguments and group access first, consolidate buffer layouts and addressing
second, then separate scheduling metadata from worker preparation state.

## Constraints

- Preserve cache keys, hash selection, mask boundaries, and transfer ordering.
- Preserve partial-block snapshots, speculative rollback, Mamba null blocks,
  pipeline layer coordinates, and request/store-job identities.
- Keep string-key fast paths and lazy grouped hashes.
- Keep `token_len_chunk`, `save_end_token`, and `target_token_len` distinct.
- Do not change configuration, dependencies, model runners, or store protocols.
- Work in the existing checkout on `codex/metadata-refactor`, as requested.
- Run affected CPU tests and repository formatting checks; report hardware
  validation separately. This internal refactor needs no release fragment.

## Approach

Moving code alone leaves the state coupling intact. Replacing all transfer
interfaces at once expands the regression surface across backends. Instead,
make three independently verifiable changes with existing compatibility entry
points where needed.

1. Normalize group access at the request boundary and remove unused parameters.
2. Move the token database out of metadata, represent each group's memory layout
   together, and share segmented address calculation between bulk and layer
   selection. Normalize broadcast descriptors during registration.
3. Make save decisions independently testable, keep tracker updates explicit,
   and move worker arrays, keys, masks, and offsets into transfer preparation
   state. Migrate production consumers together with their affected fixtures.

Metadata retains scheduling and connector payloads. Existing backend-specific
key builders retain their wire formats. No generic backend framework is added.

## Verification

Use existing metadata, scheduler, worker, snapshot, segmented-copy, hybrid,
layerwise, and PP key tests. Add focused tests for group selection, descriptor
addressing, decision side effects, and independent preparation state. Compare
literal outputs and real CPU memory copies rather than mock implementation
details. Run formatting and inspect its diff before finishing.
