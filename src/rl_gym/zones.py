"""H3 zone lookup for the rolling statistics and the request block (P2.5).

Two questions, both O(1) after construction:

- `cell_of_node(node)` -- which resolution-7 H3 cell a network node sits in, from the `h3_id`
  column of `base/nodes.csv`.
- `zone_set(cell)` -- the cell **plus its neighbours**, which is §5.2's "zone set": the unit
  P2.7 aggregates rolling counts over and P2.9 reads them from.

The neighbour relation comes from `data/rl_gym/example_network/h3_neighbours.csv`, built offline
by `src/rl_gym/offline/build_zone_table.py`. The table carries a `(cell, cell)` row for every
cell, so `zone_set` is a plain lookup rather than a lookup plus a manual re-insert.

**`h3` is not imported, here or anywhere under `src/`** (§10). The cell ids are data: they come
from `nodes.csv` and the committed table, and `h3` is deliberately not installed.

**Both lookups raise on an unknown key.** Returning an empty set instead would zero out every
zone count for that request while the episode carried on reporting healthy numbers -- P2.7 keys
its events by the event rider's origin cell, so a silent miss there is indistinguishable from a
quiet zone. An unknown node or an unknown cell means the network, the table or the caller is
wrong, and all three are worth stopping for.

The zones are an RL-internal lookup. Registering H3 as a FleetPy zone system is out of scope
(§10) and only becomes relevant when repositioning goes on, after Phase 2.
"""

import csv
import os
from typing import Dict, FrozenSet, Optional

DEFAULT_NETWORK = "example_network"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_nodes_path(network_name: str = DEFAULT_NETWORK) -> str:
    return os.path.join(_repo_root(), "data", "networks", network_name, "base", "nodes.csv")


def default_table_path(network_name: str = DEFAULT_NETWORK) -> str:
    return os.path.join(_repo_root(), "data", "rl_gym", network_name, "h3_neighbours.csv")


class ZoneTable:
    """The node -> cell map and the cell -> zone-set map, loaded once."""

    def __init__(self, nodes_path: Optional[str] = None, table_path: Optional[str] = None,
                 network_name: str = DEFAULT_NETWORK):
        """
        :param nodes_path: `base/nodes.csv`; defaults to `network_name`'s.
        :param table_path: the neighbour table; defaults to `network_name`'s.
        """
        self.nodes_path = nodes_path or default_nodes_path(network_name)
        self.table_path = table_path or default_table_path(network_name)
        self._node_cell: Dict[int, str] = self._load_node_cells(self.nodes_path)
        self._zone_set: Dict[str, FrozenSet[str]] = self._load_zone_sets(self.table_path)

        # every cell a node can be in must have a zone set, or some request's counts would be
        # unanswerable at runtime rather than at construction
        missing = sorted(set(self._node_cell.values()) - set(self._zone_set))
        if missing:
            raise ValueError(
                f"{self.table_path} has no rows for {len(missing)} cell(s) that nodes sit in: "
                f"{missing[:5]}{' …' if len(missing) > 5 else ''}"
            )

    # ---------------------------------------------------------------- #
    # the two queries
    # ---------------------------------------------------------------- #

    def cell_of_node(self, node) -> str:
        """The H3 cell node `node` sits in. Raises if the node is not in the network."""
        try:
            return self._node_cell[int(node)]
        except (KeyError, TypeError, ValueError):
            raise KeyError(
                f"node {node!r} is not in {os.path.basename(self.nodes_path)} "
                f"({len(self._node_cell)} nodes)"
            ) from None

    def zone_set(self, cell: str) -> FrozenSet[str]:
        """`cell` plus its neighbours (§5.2). Raises if `cell` is not one of the table's."""
        try:
            return self._zone_set[cell]
        except KeyError:
            raise KeyError(
                f"cell {cell!r} is not in {os.path.basename(self.table_path)} "
                f"({len(self._zone_set)} cells)"
            ) from None

    def zone_set_of_node(self, node) -> FrozenSet[str]:
        """Convenience: `zone_set(cell_of_node(node))`, the form P2.7 and P2.9 want."""
        return self.zone_set(self.cell_of_node(node))

    # ---------------------------------------------------------------- #
    # introspection, for tests and reports
    # ---------------------------------------------------------------- #

    @property
    def cells(self) -> FrozenSet[str]:
        return frozenset(self._zone_set)

    @property
    def n_nodes(self) -> int:
        return len(self._node_cell)

    def neighbours(self, cell: str) -> FrozenSet[str]:
        """`zone_set(cell)` without the cell itself."""
        return self.zone_set(cell) - {cell}

    # ---------------------------------------------------------------- #
    # loading
    # ---------------------------------------------------------------- #

    @staticmethod
    def _load_node_cells(path: str) -> Dict[int, str]:
        import pandas as pd
        if not os.path.isfile(path):
            raise ValueError(f"no node file at {path}")
        df = pd.read_csv(path, usecols=["node_index", "h3_id"])
        if df["h3_id"].isna().any():
            raise ValueError(f"{path} has null h3_id values")
        if not df["node_index"].is_unique:
            raise ValueError(f"{path} has duplicate node_index values")
        return {int(n): str(c) for n, c in zip(df["node_index"], df["h3_id"])}

    @staticmethod
    def _load_zone_sets(path: str) -> Dict[str, FrozenSet[str]]:
        if not os.path.isfile(path):
            raise ValueError(
                f"no zone table at {path}; build it with "
                f"`python -m src.rl_gym.offline.build_zone_table`"
            )
        grouped: Dict[str, set] = {}
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != ["cell", "neighbour"]:
                raise ValueError(
                    f"{path}: expected header cell,neighbour, got {reader.fieldnames}")
            for row in reader:
                grouped.setdefault(row["cell"], set()).add(row["neighbour"])
        if not grouped:
            raise ValueError(f"{path} has no rows")
        # the table is written symmetrically and reflexively; a hand-edited one might not be,
        # and a one-way row would make a zone set quietly asymmetric
        for cell, members in grouped.items():
            if cell not in members:
                raise ValueError(f"{path}: no ({cell}, {cell}) self row; zone_set would omit "
                                 f"the cell itself")
        for cell, members in grouped.items():
            for other in members:
                if other == cell:
                    continue
                if other not in grouped or cell not in grouped[other]:
                    raise ValueError(
                        f"{path}: ({cell}, {other}) is present but ({other}, {cell}) is not; "
                        f"the neighbour relation must be symmetric"
                    )
        return {cell: frozenset(members) for cell, members in grouped.items()}
