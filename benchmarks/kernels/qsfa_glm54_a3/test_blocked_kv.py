# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the 128-token QSFA KV workspace on CPU without importing TileLang."""

import unittest

from qsfa_native_reference import qsfa_glm54_mtp5_blocked_kv
from qsfa_native_reference_cube import make_cube_stages
from qsfa_native_reference_gather import make_gather_stage

TILE_TOKENS = 512
PANEL_TOKENS = 128
C0 = 16
LATENT_GROUPS = 32
ROPE_GROUPS = 4
GROUPS = LATENT_GROUPS + ROPE_GROUPS
CHANNELS = GROUPS * C0


class _Buffer:
    def access_ptr(self, mode):
        return mode


class _SourceCapture:
    """Execute the Python macros and record the source they pass to TileLang."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def macro(function):
        return function

    @staticmethod
    def alloc_var(_dtype, value):
        return value

    @staticmethod
    def Cast(_dtype, value):
        return value

    def _src_code(self, source, *arguments):
        self.calls.append((source, arguments))


def _offset(token, group):
    panel, row = divmod(token, PANEL_TOKENS)
    return ((panel * GROUPS + group) * PANEL_TOKENS + row) * C0


class BlockedKVTest(unittest.TestCase):
    def test_gather_emits_blocked_writes_in_both_metadata_modes(self):
        self.assertTrue(qsfa_glm54_mtp5_blocked_kv.keywords["blocked_kv"])
        self.assertTrue(qsfa_glm54_mtp5_blocked_kv.uses_combined_query)
        buffer = _Buffer()
        for bulk_metadata in (False, True):
            with self.subTest(bulk_metadata=bulk_metadata):
                capture = _SourceCapture()
                gather = make_gather_stage(capture, bulk_metadata=bulk_metadata, blocked_kv=True)
                gather(*([buffer] * 6), 1, 0, 2, 0, 1, TILE_TOKENS, *([buffer] * 8))
                ((source, _),) = capture.calls
                self.assertEqual("auto staged_indices" in source, bulk_metadata)
                for fragment in (
                    "constexpr int panel_tokens = 128;",
                    "const int token = start + written;",
                    "const int first_rows = rows < available ? rows : available;",
                    "const int next_rows = rows - first_rows;",
                    "const int64_t panel_base = decoded_base + (token / panel_tokens) * panel_elements;",
                    "const int64_t first_base = panel_base + panel_row * nz_lanes;",
                    "const int64_t next_base = panel_base + panel_elements;",
                    "static_cast<uint32_t>(next_rows), static_cast<uint32_t>((panel_tokens - first_rows) * 32)",
                    "static_cast<uint32_t>(first_rows), static_cast<uint32_t>((panel_tokens - next_rows) * 32)",
                    "DataCopyPad(kv_nz[first_base], nope_nz, first_copy);",
                    "DataCopyPad(kv_nz[next_base], nope_nz[first_rows * nz_lanes], next_copy);",
                    "constexpr int rope_offset = panel_tokens * quant_channels;",
                    "DataCopyPad(kv_nz[first_base + rope_offset], rope_nz, first_copy);",
                    "DataCopyPad(kv_nz[next_base + rope_offset], rope_nz[first_rows * nz_lanes], next_copy);",
                    "const DataCopyExtParams latent_zero{32, 32, 0, (panel_tokens - 1) * 32, 0};",
                    "const DataCopyExtParams rope_zero{rope_channels / nz_lanes, 32, 0, (panel_tokens - 1) * 32, 0};",
                    "for (int token = start + loaded; token < stop; ++token)",
                    "(token % panel_tokens) * nz_lanes;",
                    "DataCopyPad(kv_nz[base], zero, latent_zero);",
                    "DataCopyPad(kv_nz[base + panel_tokens * quant_channels], zero, rope_zero);",
                ):
                    self.assertTrue(fragment in source, f"missing Gather source fragment: {fragment}")
                self.assertNotIn("@STORE_", source)
                self.assertNotIn("@ZERO_FILL@", source)

    def test_cube_emits_matching_blocked_offsets(self):
        capture = _SourceCapture()
        qk, pv = make_cube_stages(capture, blocked_kv=True)
        buffer = _Buffer()
        core, slot = 3, 2
        qk(buffer, buffer, buffer, core, 5, slot, 0, TILE_TOKENS)
        pv(buffer, buffer, buffer, core, slot, 0, TILE_TOKENS)
        (qk_source, qk_args), (pv_source, pv_args) = capture.calls
        slot_base = (core * 4 + slot) * TILE_TOKENS * CHANNELS
        self.assertEqual(qk_args[4], slot_base)
        self.assertEqual(pv_args[3], slot_base)
        self.assertIn("n0 * qsfa_combined + k1 * qk_l1_k * qsfa_n_panel", qk_source)
        self.assertIn("k_copy.srcStride = qsfa_n_panel - n;", qk_source)
        self.assertIn("kv_nz[k_offset + 224 * qsfa_n_panel]", qk_source)
        self.assertIn("(k1 + k0) * qsfa_combined + n0 * qsfa_n_panel", pv_source)
        self.assertIn("v_copy.srcStride = qsfa_n_panel - k;", pv_source)
        self.assertIn("p_copy.srcDValue = qsfa_s2;", pv_source)
        self.assertIn("fix.dstStride = qsfa_s2;", qk_source)
        self.assertIn("fix.dstStride = qsfa_s2;", pv_source)

    def test_gather_flushes_preserve_ub_pitch_across_every_boundary(self):
        for start in range(TILE_TOKENS):
            for rows in range(1, min(32, TILE_TOKENS - start) + 1):
                first = min(rows, PANEL_TOKENS - start % PANEL_TOKENS)
                for copied, length in ((0, first), (first, rows - first)):
                    if not length:
                        continue
                    token = start + copied
                    panel_base = token // PANEL_TOKENS * PANEL_TOKENS * CHANNELS
                    gm_start = panel_base + token % PANEL_TOKENS * C0
                    ub_gap = rows - length  # UB gap uses 32-byte blocks.
                    gm_gap = (PANEL_TOKENS - length) * 32  # GM gap uses bytes.
                    for group in range(GROUPS):
                        actual_ub = copied * C0 + group * (length + ub_gap) * C0
                        actual_gm = gm_start + group * (length * 32 + gm_gap) // 2
                        assert actual_ub == (group * rows + copied) * C0, (start, rows, group)
                        assert actual_gm == _offset(token, group), (start, rows, group)
                        assert actual_gm + (length - 1) * C0 == _offset(token + length - 1, group), (
                            start,
                            rows,
                            group,
                        )

    def test_cube_reads_each_gathered_cell_once_per_consumer(self):
        workspace = {_offset(token, group): (token, group) for token in range(TILE_TOKENS) for group in range(GROUPS)}
        self.assertEqual(set(workspace), set(range(0, TILE_TOKENS * CHANNELS, C0)))
        for width in (1, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512):
            for token_base in range(0, width, PANEL_TOKENS):
                tokens = min(width - token_base, PANEL_TOKENS)
                for copy_groups in (((0, 18), (18, 14), (32, 4)), ((0, 8), (8, 8), (16, 8), (24, 8))):
                    for first_group, groups in copy_groups:
                        base = token_base * CHANNELS + first_group * PANEL_TOKENS * C0
                        source_stride = PANEL_TOKENS - tokens
                        for group in range(groups):
                            source = base + group * (tokens + source_stride) * C0
                            actual = [workspace[source + row * C0] for row in range(tokens)]
                            expected = [(token_base + row, first_group + group) for row in range(tokens)]
                            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
