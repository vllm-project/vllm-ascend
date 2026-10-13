"""CPU checks for AIV offset coalescing/pairing and Cube copy replay."""
import random
import unittest


def describe(indices, y, start, size, n, stride, pages=None, ks=0, vs=0):
    groups = []
    for begin in range(0, size, n):
        compact = start + begin
        group_size = min(n, size - begin)
        inner, done, entries = compact % y, 0, []
        block = compact // y
        while done < group_size:
            rows = min(y - inner, group_size - done)
            token = indices[block] * y + inner
            copied = 0
            while copied < rows:
                t, length = token + copied, rows - copied
                if pages is None:
                    k = v = t * stride
                else:
                    length = min(length, 128 - t % 128)
                    k = pages[t // 128] * ks + t % 128 * stride
                    v = pages[t // 128] * vs + t % 128 * stride
                entries.append((k, v, length))
                copied += length
            done += rows
            block += 1
            inner = 0
        groups.append(entries)
    return groups


def prepare(group, y, paged, stride=None):
    if paged:
        return group
    records = []
    for offset, _, rows in group:
        if records:
            previous, length, num, _ = records[-1]
            step = offset - previous
            if num == 1 and stride is not None and step == length * stride:
                records[-1] = (previous, length + rows, 1, 0)
                continue
            if num == 1 and length == rows == y and 0 <= step <= 0xffffffff and step % 16 == 0:
                records[-1] = (previous, length, 2, step)
                continue
        records.append((offset, rows, 1, 0))
    return records


def replay(group, stride, paged, field):
    result, pairs = [], 0
    for record in group:
        if paged:
            offset, rows, num, step = record[field], record[2], 1, 0
        else:
            offset, rows, num, step = record
        pairs += num == 2
        for j in range(num):
            result += [offset + j * step + row * stride for row in range(rows)]
    return result, pairs


class SparseOffsetTest(unittest.TestCase):
    def test_geometry(self):
        rng = random.Random(20260924)
        pairs = 0
        for y in range(1, 129):
            for tile in (128, 256, 384, 512):
                for n in (128, 256):
                    for start in (0, y - 1, tile):
                        indices = list(range(0, 4 * tile // y + 12, 2))
                        tokens = [i * y + j for i in indices for j in range(y)][start:start + tile - 1]
                        stride = (33, 128, 256, 512)[y % 4] * 3
                        pages = list(range(max(tokens) // 128 + 1))
                        rng.shuffle(pages)
                        ks, vs = 128 * stride * 2, 128 * stride * 3
                        for paged in (False, True):
                            groups = describe(indices, y, start, len(tokens), n, stride,
                                              pages if paged else None, ks, vs)
                            capacity = (n + 2 * y - 2) // y
                            if paged:
                                capacity = min(n, 2 * capacity)
                            for field, ps in enumerate((ks, vs)):
                                result = []
                                for g, group in enumerate(groups):
                                    self.assertEqual(sum(row[2] for row in group), min(n, len(tokens) - g * n))
                                    records = prepare(group, y, paged, stride)
                                    self.assertLessEqual(len(records), capacity)
                                    rows, paired = replay(records, stride, paged, field)
                                    pairs += paired
                                    if paged:
                                        self.assertEqual(paired, 0)
                                    result += rows
                                expected = [pages[t // 128] * ps + t % 128 * stride if paged else t * stride
                                            for t in tokens]
                                self.assertEqual(result, expected)
        self.assertGreater(pairs, 0)

    def test_page_split_count(self):
        groups = describe([1], 127, 0, 127, 128, 128, [7, 3], 16384, 32768)
        self.assertEqual(groups, [[(130944, 245632, 1), (49152, 98304, 126)]])
        self.assertEqual(describe([1], 127, 0, 1, 128, 128, [7], 16384, 32768),
                         [[(130944, 245632, 1)]])

    def test_contiguous_merge(self):
        for y in (1, 3, 16, 31, 128):
            for stride in (33, 128, 512):
                groups = describe([0, 1, 2, 4, 6, 7], y, 0, 6 * y - 1, 256, stride)
                expected = [block * y * stride + row * stride for block in (0, 1, 2, 4, 6, 7)
                            for row in range(y)][:-1]
                actual = []
                for group in groups:
                    actual += replay(prepare(group, y, False, stride), stride, False, 0)[0]
                self.assertEqual(actual, expected)
        self.assertEqual(prepare([(0, 0, 16), (2048, 2048, 16), (4096, 4096, 3)],
                                 16, False, 128), [(0, 35, 1, 0)])

    def test_random_runs(self):
        rng = random.Random(20260925)
        for y in range(1, 129):
            for stride in (1, 33, 128, 1024):
                indices, last = [], 0
                for _ in range(520):
                    last += rng.choice((1, 1, 1, 2, 3))
                    indices.append(last)
                for n in (128, 256):
                    start, size = rng.randrange(y), 511
                    tokens = [i * y + row for i in indices for row in range(y)][start:start + size]
                    groups = describe(indices, y, start, size, n, stride)
                    actual = []
                    for group in groups:
                        actual += replay(prepare(group, y, False, stride), stride, False, 0)[0]
                    self.assertEqual(actual, [t * stride for t in tokens])

    def test_large_stride_single(self):
        records = prepare([(0, 0, 16), (1 << 40, 1 << 40, 16)], 16, False)
        self.assertEqual(replay(records, 128, False, 0)[1], 0)

    def test_workspace(self):
        for paged, expected in ((False, 140 * 1024), (True, 196 * 1024)):
            blocks = (256 + 2 * 16 - 2) // 16
            if paged:
                blocks = min(256, 2 * blocks)
            group_bytes = (64 + blocks * (24 if paged else 32) + 63) // 64 * 64
            self.assertEqual(28 * 4 * 2 * group_bytes, expected)

    def test_ub_all_dimensions(self):
        for d in range(1, 513):
            embed = (d + 31) // 32 * 32
            for q in (16, 32, 64, 128):
                qr = max(q // 2, 16)
                for precision in (0, 1):
                    tile = (512 if q <= 32 else 256 if q <= 64 else 128) if d <= 256 else (
                        384 if q <= 16 else 256 if q <= 32 else 128)
                    if precision:
                        tile = min(tile, 256)
                    n = 256 if d <= 128 and q <= 64 else 128
                    transit = d > 256 and q > 64
                    lo_bufs = 1 if not transit and d > 128 and q > 64 else 2
                    sp = qr * tile * (4 + 2) * 2
                    go = qr * embed * 4
                    mask_start = sp + go + (0 if transit else lo_bufs * go) + 3840
                    mask_bytes = qr * tile
                    for y in range(1, 129):
                        cap = (n + 2 * y - 2) // y
                        for paged in (False, True):
                            count = min(n, 2 * cap) if paged else cap
                            table_bytes = (64 + count * (24 if paged else 32) + 63) // 64 * 64
                            self.assertLessEqual(mask_start + table_bytes, 248 * 1024)
                            if transit:
                                old_rows = (248 * 1024 - mask_start - mask_bytes) // (embed * 4 * lo_bufs)
                                new_rows = (248 * 1024 - mask_start - max(mask_bytes, table_bytes)) // (embed * 4 * lo_bufs)
                                self.assertEqual(new_rows, old_rows)
                                self.assertGreater(new_rows, 0)


if __name__ == "__main__":
    unittest.main()
