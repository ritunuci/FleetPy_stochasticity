"""Build the H3 neighbour table for `example_network` (P2.5).

The zones are the **37 resolution-7 H3 cells** that the network's nodes fall in -- the distinct
values of the `h3_id` column of `base/nodes.csv`. Two cells are neighbours when their polygons
in `h3_hexagons.geojson` touch, which in a hexagonal tiling is edge-sharing.

**The touching test is exact, not a tolerance.** Measured over all 666 cell pairs: the 88
neighbour pairs are at shapely distance **exactly 0.0** -- adjacent hexagons in this geojson
share identical vertex coordinates -- and the nearest **non**-neighbour pair is at 1.377e-02
degrees. Every threshold from 0 to 1e-2 yields the same 88 pairs, so `TOUCH_TOLERANCE` sits in a
gap seven orders of magnitude wide on either side. Do not "tighten" or "loosen" it thinking it
is delicate; it is not doing any rounding.

**Nothing imports `h3`** (§10). The cell ids and the polygons are both already in the repository,
and `h3` is deliberately not installed. The figures this reproduces were computed during drafting
with `h3` 4.5.0 `grid_disk(cell, 1)` restricted to the 37 cells: 88 undirected pairs, every cell
with 2-6 neighbours inside the set, 17 cells with all 6.

    python -m src.rl_gym.offline.build_zone_table            # write the table
    python -m src.rl_gym.offline.build_zone_table --check    # verify the committed table

`--check` compares the **parsed `(cell, neighbour)` pairs as sets**, not file bytes, so a
line-ending difference on another machine cannot fail it.

Output: `data/rl_gym/example_network/h3_neighbours.csv`, `cell,neighbour`, sorted by both
columns, **including the 37 `(cell, cell)` self rows** -- 176 directed + 37 self = 213 rows. The
self rows are what make `zone_set(cell)` the cell *plus* its neighbours (§5.2) without the
reader having to add the cell back.
"""

import argparse
import csv
import itertools
import json
import os
import sys
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NETWORK = "example_network"
GEOJSON = os.path.join(ROOT, "data", "networks", NETWORK, "h3_hexagons.geojson")
NODES = os.path.join(ROOT, "data", "networks", NETWORK, "base", "nodes.csv")
TABLE = os.path.join(ROOT, "data", "rl_gym", NETWORK, "h3_neighbours.csv")

#: Polygons closer than this are neighbours. See the module docstring: the real separation is
#: 0.0 for neighbours and 1.377e-02 for everything else, so this is not a rounding decision.
TOUCH_TOLERANCE = 1e-9

#: figures from the drafting-time `h3` 4.5.0 computation, asserted on every build
EXPECTED_CELLS = 37
EXPECTED_PAIRS = 88
EXPECTED_FULL_DEGREE = 17        # cells with all 6 neighbours inside the set
EXPECTED_DEGREE_RANGE = (2, 6)


def load_cell_polygons(geojson_path: str = GEOJSON) -> Dict[str, "object"]:
    from shapely.geometry import shape
    with open(geojson_path) as fh:
        g = json.load(fh)
    polys = {f["properties"]["h3_id"]: shape(f["geometry"]) for f in g["features"]}
    if len(polys) != len(g["features"]):
        raise ValueError("duplicate h3_id in the geojson")
    invalid = sorted(c for c, p in polys.items() if not p.is_valid)
    if invalid:
        raise ValueError(f"invalid polygons: {invalid}")
    return polys


def node_cells(nodes_path: str = NODES) -> Dict[int, str]:
    """`node_index -> h3_id` from `nodes.csv`."""
    import pandas as pd
    df = pd.read_csv(nodes_path)
    for col in ("node_index", "h3_id"):
        if col not in df.columns:
            raise ValueError(f"{nodes_path} has no {col} column")
    if df["h3_id"].isna().any():
        raise ValueError(f"{nodes_path} has null h3_id values")
    if not df["node_index"].is_unique:
        raise ValueError(f"{nodes_path} has duplicate node_index values")
    return {int(n): str(c) for n, c in zip(df["node_index"], df["h3_id"])}


def neighbour_pairs(polys: Dict[str, "object"]) -> Set[Tuple[str, str]]:
    """Undirected `(a, b)` with `a < b` for every pair of touching cells."""
    cells = sorted(polys)
    return {(a, b) for a, b in itertools.combinations(cells, 2)
            if polys[a].distance(polys[b]) < TOUCH_TOLERANCE}


def build_rows(polys: Dict[str, "object"]) -> List[Tuple[str, str]]:
    """Every `(cell, neighbour)` row, self rows included, sorted by both columns."""
    cells = sorted(polys)
    rows: Set[Tuple[str, str]] = {(c, c) for c in cells}       # reflexive, so zone_set is cell+ring
    for a, b in neighbour_pairs(polys):
        rows.add((a, b))
        rows.add((b, a))                                        # symmetric, written both ways
    return sorted(rows)


def check_invariants(polys: Dict[str, "object"], node_cell_map: Dict[int, str]) -> None:
    """Refuse to write unless the table matches the drafting-time `h3` figures."""
    cells = set(polys)
    if len(cells) != EXPECTED_CELLS:
        raise ValueError(f"expected {EXPECTED_CELLS} cells in the geojson, found {len(cells)}")
    from_nodes = set(node_cell_map.values())
    if from_nodes != cells:
        raise ValueError(
            f"the geojson's cells and nodes.csv's h3_id values disagree: "
            f"{sorted(cells - from_nodes)} only in the geojson, "
            f"{sorted(from_nodes - cells)} only in nodes.csv"
        )
    pairs = neighbour_pairs(polys)
    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError(f"expected {EXPECTED_PAIRS} undirected pairs, found {len(pairs)}")
    degree = {c: 0 for c in cells}
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1
    lo, hi = min(degree.values()), max(degree.values())
    if (lo, hi) != EXPECTED_DEGREE_RANGE:
        raise ValueError(f"expected degrees in {EXPECTED_DEGREE_RANGE}, found {(lo, hi)}")
    full = sum(1 for d in degree.values() if d == 6)
    if full != EXPECTED_FULL_DEGREE:
        raise ValueError(f"expected {EXPECTED_FULL_DEGREE} cells with 6 neighbours, found {full}")


def read_table(path: str = TABLE) -> Set[Tuple[str, str]]:
    """The committed table as a set of `(cell, neighbour)` pairs.

    Parsed, not read as bytes: `--check` must survive a line-ending difference on another
    machine, exactly as §2.9's corpus twins do.
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["cell", "neighbour"]:
            raise ValueError(f"{path}: expected header cell,neighbour, got {reader.fieldnames}")
        return {(row["cell"], row["neighbour"]) for row in reader}


def write_table(rows: List[Tuple[str, str]], path: str = TABLE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell", "neighbour"])
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the H3 neighbour table (P2.5)")
    ap.add_argument("--check", action="store_true",
                    help="verify the committed table instead of writing it; compares parsed "
                         "(cell, neighbour) pairs, not file bytes")
    ap.add_argument("--table", default=TABLE)
    args = ap.parse_args()

    polys = load_cell_polygons()
    node_cell_map = node_cells()
    check_invariants(polys, node_cell_map)
    rows = build_rows(polys)
    pairs = neighbour_pairs(polys)

    print(f"cells {len(polys)} | undirected pairs {len(pairs)} | rows {len(rows)} "
          f"({2 * len(pairs)} directed + {len(polys)} self)")
    print(f"nodes mapped {len(node_cell_map)} | touch tolerance {TOUCH_TOLERANCE:g}")

    if not args.check:
        write_table(rows, args.table)
        print(f"wrote {args.table}")
        return 0

    committed = read_table(args.table)
    built = set(rows)
    missing = sorted(built - committed)
    extra = sorted(committed - built)
    print(f"committed rows {len(committed)} | rebuilt rows {len(built)}")
    if missing or extra:
        print(f"MISMATCH: {len(missing)} missing, {len(extra)} extra")
        for a, b in missing[:10]:
            print(f"  missing {a},{b}")
        for a, b in extra[:10]:
            print(f"  extra   {a},{b}")
        return 1
    print("the committed table matches a fresh build, pair for pair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
