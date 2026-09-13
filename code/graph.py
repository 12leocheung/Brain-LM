"""
Associative memory graph — core data structure.

Nodes: {id, type, data}
Edges: weighted, undirected-in-effect (stored symmetrically), representing
       "pathway thickness" between two memories.

Update rule: co-retrieval Hebbian — whenever two memories are retrieved
together in the same query/response cycle, their edge weight increments.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryNode:
    id: str
    type: str
    data: Any

    def __repr__(self):
        return f"<{self.type}:{self.id} '{self.data}'>"


class MemoryGraph:
    def __init__(self, increment: float = 1.0, decay: float = 0.0, max_weight: float | None = None):
        """
        increment:  amount added to an edge weight on each co-retrieval event.
        decay:      multiplicative decay applied to ALL edges each time step()
                    is called (0.0 = no decay, i.e. weights never shrink).
        max_weight: optional cap on edge weight (None = unbounded).
        """
        self.nodes: dict[str, MemoryNode] = {}
        # edges[a][b] = weight, kept symmetric: edges[a][b] == edges[b][a]
        self.edges: dict[str, dict[str, float]] = {}
        self.increment = increment
        self.decay = decay
        self.max_weight = max_weight

    # ---------- node / edge management ----------

    def add_node(self, id: str, type: str, data: Any) -> MemoryNode:
        node = MemoryNode(id=id, type=type, data=data)
        self.nodes[id] = node
        self.edges.setdefault(id, {})
        return node

    def add_edge(self, a: str, b: str, weight: float = 0.0):
        """Create (or overwrite) a direct edge between two existing nodes."""
        assert a in self.nodes and b in self.nodes, "both nodes must exist first"
        self.edges.setdefault(a, {})[b] = weight
        self.edges.setdefault(b, {})[a] = weight

    def neighbors(self, node_id: str) -> dict[str, float]:
        return self.edges.get(node_id, {})

    # ---------- Hebbian co-retrieval update ----------

    def reinforce(self, retrieved_ids: list[str]):
        """
        Call this once per query/response cycle with the set of memory ids
        that were retrieved together. Every pair in that set gets its edge
        strengthened (or created, if it didn't exist yet).
        """
        for i in range(len(retrieved_ids)):
            for j in range(i + 1, len(retrieved_ids)):
                a, b = retrieved_ids[i], retrieved_ids[j]
                if a == b:
                    continue
                current = self.edges.get(a, {}).get(b, 0.0)
                new_weight = current + self.increment
                if self.max_weight is not None:
                    new_weight = min(new_weight, self.max_weight)
                self.add_edge(a, b, new_weight)

    def step(self):
        """
        Apply decay to every edge. Call this periodically (e.g. once per
        query cycle, or on a slower clock) if you want unused pathways to
        weaken over time. With decay=0.0 this is a no-op.
        """
        if self.decay <= 0.0:
            return
        for a, nbrs in self.edges.items():
            for b in list(nbrs.keys()):
                nbrs[b] *= (1.0 - self.decay)

    # ---------- inspection helpers ----------

    def describe(self):
        print(f"{len(self.nodes)} nodes, "
              f"{sum(len(v) for v in self.edges.values()) // 2} edges")
        for a, nbrs in self.edges.items():
            for b, w in nbrs.items():
                if a < b:  # print each undirected edge once
                    print(f"  {a} --{w:.2f}-- {b}")
