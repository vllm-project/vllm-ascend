import unittest

from analyze_expert_cache_log import parse_lines, summarize


class AnalyzeExpertCacheLogTest(unittest.TestCase):

    def test_summarizes_layer_step_and_global_hit_rates(self):
        lines = [
            "[UPDATE-W] l=0 call=10 topk_shape=(1, 6) |needed|=6 |on_dev|=12 |to_load|=4 reusable=10 needed=[1, 2, 3, 4, 5, 6]",
            "[UPDATE-W] l=0 cache_hit=[1, 2] cache_miss=[3, 4, 5, 6]",
            "[UPDATE-W] l=1 call=11 topk_shape=(1, 6) |needed|=6 |on_dev|=12 |to_load|=0 reusable=6 needed=[7, 8, 9, 10, 11, 12]",
            "[UPDATE-W] l=1 cache_hit=[7, 8, 9, 10, 11, 12] cache_miss=[]",
            "[UPDATE-W] l=0 call=12 topk_shape=(1, 6) |needed|=6 |on_dev|=12 |to_load|=3 reusable=9 needed=[1, 2, 3, 7, 8, 9]",
            "[UPDATE-W] l=0 cache_hit=[1, 2, 3] cache_miss=[7, 8, 9]",
        ]

        summary = summarize(parse_lines(lines), decode_step_layers=2)

        self.assertEqual(summary.global_hits, 11)
        self.assertEqual(summary.global_requests, 18)
        self.assertAlmostEqual(summary.global_hit_rate, 11 / 18)
        self.assertAlmostEqual(summary.layers[0].hit_rate, 5 / 12)
        self.assertAlmostEqual(summary.layers[1].hit_rate, 1.0)
        self.assertEqual(summary.event_steps[10].hit_rate, 2 / 6)
        self.assertEqual(summary.event_steps[11].hit_rate, 1.0)
        self.assertEqual(summary.event_steps[12].hit_rate, 3 / 6)
        self.assertAlmostEqual(summary.decode_steps[5].hit_rate, 8 / 12)
        self.assertAlmostEqual(summary.decode_steps[6].hit_rate, 3 / 6)


if __name__ == "__main__":
    unittest.main()
