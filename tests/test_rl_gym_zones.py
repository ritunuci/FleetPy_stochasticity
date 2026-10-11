"""Tests for src/rl_gym/zones.py and the committed H3 neighbour table (P2.5).

`python -m unittest discover tests`. Fast: no FleetPy simulation is built.

The figures asserted here were computed during drafting with `h3` 4.5.0 `grid_disk(cell, 1)`
restricted to the 37 cells. **`h3` is not installed and must not be** (§0, §10), so they are
compared against as numbers rather than recomputed by calling it.

Most of these read the **committed CSV**, not the builder's in-memory output, so what gets
checked is the artifact the env will actually load.
"""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl_gym.offline.build_zone_table import (  # noqa: E402
    EXPECTED_CELLS,
    EXPECTED_FULL_DEGREE,
    EXPECTED_PAIRS,
    TOUCH_TOLERANCE,
    build_rows,
    load_cell_polygons,
    neighbour_pairs,
    node_cells,
    read_table,
)
from src.rl_gym.zones import ZoneTable, default_nodes_path, default_table_path  # noqa: E402

# the drafting-time h3 figures
N_CELLS = 37
N_UNDIRECTED_PAIRS = 88
N_FULL_DEGREE = 17              # cells with all 6 neighbours inside the set
DEGREE_HISTOGRAM = {2: 1, 3: 9, 4: 5, 5: 5, 6: 17}
N_ROWS = 2 * N_UNDIRECTED_PAIRS + N_CELLS       # 176 directed + 37 self = 213
N_NODES = 4084


class TestCommittedTable(unittest.TestCase):
    """The artifact on disk, read as the env reads it."""

    @classmethod
    def setUpClass(cls):
        cls.pairs = read_table(default_table_path())

    def test_row_count(self):
        self.assertEqual(len(self.pairs), N_ROWS)

    def test_cell_count(self):
        self.assertEqual(len({c for c, _ in self.pairs}), N_CELLS)
        self.assertEqual(len({n for _, n in self.pairs}), N_CELLS)

    def test_relation_is_symmetric(self):
        asymmetric = [(a, b) for a, b in self.pairs if (b, a) not in self.pairs]
        self.assertEqual(asymmetric, [])

    def test_relation_is_reflexive(self):
        cells = {c for c, _ in self.pairs}
        self.assertEqual(sorted(c for c in cells if (c, c) not in self.pairs), [])

    def test_undirected_pair_count(self):
        undirected = {(a, b) for a, b in self.pairs if a < b}
        self.assertEqual(len(undirected), N_UNDIRECTED_PAIRS)

    def test_degree_histogram_matches_h3(self):
        degree = {}
        for a, b in self.pairs:
            if a != b:
                degree[a] = degree.get(a, 0) + 1
        hist = {}
        for cell in {c for c, _ in self.pairs}:
            d = degree.get(cell, 0)
            hist[d] = hist.get(d, 0) + 1
        self.assertEqual(hist, DEGREE_HISTOGRAM)
        self.assertEqual(hist.get(6), N_FULL_DEGREE)
        self.assertEqual(min(hist), 2)
        self.assertEqual(max(hist), 6)

    def test_every_id_is_a_network_cell(self):
        from_nodes = set(node_cells().values())
        ids = {c for c, _ in self.pairs} | {n for _, n in self.pairs}
        self.assertEqual(ids, from_nodes)

    def test_rows_are_sorted_and_header_is_exact(self):
        with open(default_table_path(), newline="") as fh:
            reader = csv.reader(fh)
            self.assertEqual(next(reader), ["cell", "neighbour"])
            rows = [tuple(r) for r in reader]
        self.assertEqual(rows, sorted(rows), "committed rows must be sorted, for a stable diff")
        self.assertEqual(len(rows), len(set(rows)), "no duplicate rows")


class TestBuilderAgreesWithTheCommittedTable(unittest.TestCase):
    """`--check`'s comparison, as a test: parsed pairs, never bytes."""

    def test_fresh_build_matches_pair_for_pair(self):
        built = set(build_rows(load_cell_polygons()))
        committed = read_table(default_table_path())
        self.assertEqual(built, committed)

    def test_comparison_survives_crlf(self):
        # the point of comparing parsed pairs: §2.9's corpus twins differ only in line endings
        committed = read_table(default_table_path())
        with open(default_table_path(), newline="") as fh:
            text = fh.read()
        with tempfile.TemporaryDirectory() as tmp:
            crlf = os.path.join(tmp, "crlf.csv")
            with open(crlf, "w", newline="") as fh:
                fh.write(text.replace("\n", "\r\n"))
            with open(crlf, "rb") as a, open(default_table_path(), "rb") as b:
                self.assertNotEqual(a.read(), b.read(), "the two files must differ as bytes")
            self.assertEqual(read_table(crlf), committed)


class TestTouchingRuleIsExactNotApproximate(unittest.TestCase):
    """The 1e-9 threshold sits in a gap seven orders of magnitude wide (P2.5)."""

    @classmethod
    def setUpClass(cls):
        cls.polys = load_cell_polygons()

    def test_neighbours_are_at_distance_exactly_zero(self):
        for a, b in neighbour_pairs(self.polys):
            self.assertEqual(self.polys[a].distance(self.polys[b]), 0.0, f"{a},{b}")

    def test_non_neighbours_are_far_beyond_the_tolerance(self):
        import itertools
        pairs = neighbour_pairs(self.polys)
        cells = sorted(self.polys)
        apart = [self.polys[a].distance(self.polys[b])
                 for a, b in itertools.combinations(cells, 2)
                 if (a, b) not in pairs]
        self.assertGreater(min(apart), 1e-3)
        self.assertGreater(min(apart) / TOUCH_TOLERANCE, 1e6,
                           "the tolerance must not sit anywhere near a real distance")

    def test_pair_count_is_invariant_across_seven_orders_of_magnitude(self):
        import itertools
        cells = sorted(self.polys)
        dists = [self.polys[a].distance(self.polys[b])
                 for a, b in itertools.combinations(cells, 2)]
        for thr in (1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3):
            self.assertEqual(sum(1 for d in dists if d < thr), N_UNDIRECTED_PAIRS,
                             f"threshold {thr:g} changes the neighbour set")

    def test_polygon_and_cell_counts(self):
        self.assertEqual(len(self.polys), N_CELLS)
        self.assertEqual(EXPECTED_CELLS, N_CELLS)
        self.assertEqual(EXPECTED_PAIRS, N_UNDIRECTED_PAIRS)
        self.assertEqual(EXPECTED_FULL_DEGREE, N_FULL_DEGREE)


class TestZoneTable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.zt = ZoneTable()

    def test_loads_every_node(self):
        self.assertEqual(self.zt.n_nodes, N_NODES)

    def test_cells(self):
        self.assertEqual(len(self.zt.cells), N_CELLS)

    def test_cell_of_node_matches_nodes_csv(self):
        expected = node_cells()
        for node in (0, 1, 42, 2000, N_NODES - 1):
            self.assertEqual(self.zt.cell_of_node(node), expected[node])

    def test_cell_of_node_accepts_an_int_like_value(self):
        # FleetPy positions carry numpy ints and occasionally floats
        import numpy as np
        self.assertEqual(self.zt.cell_of_node(np.int64(7)), self.zt.cell_of_node(7))
        self.assertEqual(self.zt.cell_of_node(7.0), self.zt.cell_of_node(7))

    def test_zone_set_contains_the_cell_itself(self):
        for cell in sorted(self.zt.cells):
            self.assertIn(cell, self.zt.zone_set(cell))

    def test_zone_set_sizes_are_one_plus_degree(self):
        sizes = {}
        for cell in self.zt.cells:
            n = len(self.zt.zone_set(cell))
            sizes[n] = sizes.get(n, 0) + 1
        self.assertEqual(sizes, {d + 1: n for d, n in DEGREE_HISTOGRAM.items()})

    def test_zone_set_is_symmetric(self):
        for cell in sorted(self.zt.cells):
            for other in self.zt.neighbours(cell):
                self.assertIn(cell, self.zt.zone_set(other))

    def test_neighbours_excludes_self(self):
        for cell in sorted(self.zt.cells):
            self.assertNotIn(cell, self.zt.neighbours(cell))
            self.assertEqual(len(self.zt.neighbours(cell)),
                             len(self.zt.zone_set(cell)) - 1)

    def test_zone_set_of_node_composes_the_two(self):
        for node in (0, 500, 4083):
            self.assertEqual(self.zt.zone_set_of_node(node),
                             self.zt.zone_set(self.zt.cell_of_node(node)))

    def test_every_node_has_a_zone_set(self):
        # the construction-time guard, confirmed end to end: no node can be unanswerable
        for node in range(0, N_NODES, 97):
            self.assertGreaterEqual(len(self.zt.zone_set_of_node(node)), 3)

    def test_zone_set_is_immutable(self):
        s = self.zt.zone_set(next(iter(self.zt.cells)))
        self.assertIsInstance(s, frozenset)


class TestUnknownKeysRaise(unittest.TestCase):
    """Raising, not returning empty -- an empty zone set would zero counts silently (P2.7)."""

    @classmethod
    def setUpClass(cls):
        cls.zt = ZoneTable()

    def test_unknown_node_raises(self):
        for node in (-1, N_NODES, 10 ** 9):
            with self.assertRaises(KeyError) as cm:
                self.zt.cell_of_node(node)
            self.assertIn(str(node), str(cm.exception))

    def test_non_numeric_node_raises(self):
        for node in ("abc", None, object()):
            with self.assertRaises(KeyError):
                self.zt.cell_of_node(node)

    def test_unknown_cell_raises(self):
        with self.assertRaises(KeyError) as cm:
            self.zt.zone_set("872834000ffffff")
        self.assertIn("872834000ffffff", str(cm.exception))

    def test_empty_is_never_returned(self):
        for cell in self.zt.cells:
            self.assertTrue(self.zt.zone_set(cell))


class TestMalformedTablesAreRejected(unittest.TestCase):
    """A hand-edited table must fail at construction, not at the first decision."""

    def _write(self, tmp, rows, header=("cell", "neighbour")):
        path = os.path.join(tmp, "t.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(header))
            w.writerows(rows)
        return path

    def test_missing_file_raises_with_the_build_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as cm:
                ZoneTable(table_path=os.path.join(tmp, "nope.csv"))
            self.assertIn("build_zone_table", str(cm.exception))

    def test_wrong_header_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [("a", "a")], header=("from", "to"))
            with self.assertRaises(ValueError):
                ZoneTable(table_path=path)

    def test_asymmetric_table_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [("a", "a"), ("b", "b"), ("a", "b")])
            with self.assertRaises(ValueError) as cm:
                ZoneTable(table_path=path)
            self.assertIn("symmetric", str(cm.exception))

    def test_missing_self_row_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [("a", "b"), ("b", "a"), ("b", "b")])
            with self.assertRaises(ValueError) as cm:
                ZoneTable(table_path=path)
            self.assertIn("self row", str(cm.exception))

    def test_empty_table_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [])
            with self.assertRaises(ValueError):
                ZoneTable(table_path=path)

    def test_table_missing_a_cell_that_nodes_sit_in_raises(self):
        # the construction-time guard: a table that is internally consistent but does not cover
        # the network would otherwise fail at the first request in that cell
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [("8728340d2ffffff", "8728340d2ffffff")])
            with self.assertRaises(ValueError) as cm:
                ZoneTable(table_path=path)
            self.assertIn("cell(s) that nodes sit in", str(cm.exception))


class TestNothingImportsH3(unittest.TestCase):
    """§10: `h3` is not a dependency of anything under `src/`."""

    def test_h3_is_not_imported_by_zones_or_the_builder(self):
        for mod in ("src.rl_gym.zones", "src.rl_gym.offline.build_zone_table"):
            __import__(mod)
        self.assertNotIn("h3", sys.modules)

    def test_h3_is_not_installed(self):
        with self.assertRaises(ImportError):
            __import__("h3")

    def test_no_source_file_mentions_h3_as_an_import(self):
        import re
        root = os.path.join(os.path.dirname(__file__), "..", "src", "rl_gym")
        pattern = re.compile(r"^\s*(import\s+h3|from\s+h3\b)", re.M)
        offenders = []
        for dirpath, _dirs, files in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                with open(os.path.join(dirpath, name)) as fh:
                    if pattern.search(fh.read()):
                        offenders.append(os.path.join(dirpath, name))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
