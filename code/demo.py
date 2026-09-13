"""
A denser, more realistic toy graph: several loosely-clustered topics with
multiple cross-links and competing mid-strength edges, so a walk actually
has real choices to make rather than following one dominant chain.

Clusters (not enforced in code, just how the data was designed):
  - pets:      dog, bark, cat, meow, pet, leash, vet
  - outdoors:  walk, park, tree, squirrel, bird, nest, rain
  - food:      kitchen, cook, recipe, bread, oven, coffee
  - work:      office, meeting, laptop, email, coffee (shared with food)

Cross-links deliberately connect clusters at more than one point, and with
varied weights, so no single "spine" dominates every walk.
"""

import random
from graph import MemoryGraph
from walk import random_walk, multi_walk

rng = random.Random(7)

g = MemoryGraph(increment=0.5, decay=0.03, max_weight=10.0)

nodes = [
    # pets cluster
    "dog", "bark", "cat", "meow", "pet", "leash", "vet",
    # outdoors cluster
    "walk", "park", "tree", "squirrel", "bird", "nest", "rain",
    # food cluster
    "kitchen", "cook", "recipe", "bread", "oven", "coffee",
    # work cluster
    "office", "meeting", "laptop", "email",
]
for n in nodes:
    g.add_node(n, "association", n)

edges = [
    # pets cluster (varied weights, more than one hub)
    ("dog", "bark", 4.0), ("dog", "pet", 3.0), ("dog", "leash", 3.5),
    ("cat", "meow", 4.5), ("cat", "pet", 2.5), ("pet", "vet", 2.0),
    ("dog", "vet", 1.5), ("cat", "leash", 0.5),

    # outdoors cluster
    ("leash", "walk", 4.0), ("walk", "park", 3.5), ("park", "tree", 2.5),
    ("tree", "squirrel", 2.0), ("tree", "bird", 2.5), ("bird", "nest", 3.0),
    ("park", "rain", 1.0), ("walk", "rain", 1.5),

    # food cluster
    ("kitchen", "cook", 3.5), ("cook", "recipe", 3.0), ("cook", "bread", 2.5),
    ("bread", "oven", 3.5), ("kitchen", "coffee", 2.0), ("recipe", "oven", 1.5),

    # work cluster
    ("office", "meeting", 3.0), ("office", "laptop", 3.5), ("laptop", "email", 3.0),
    ("meeting", "email", 2.0), ("office", "coffee", 2.5),

    # cross-cluster links (deliberately more than one bridge, so the walk
    # has a real choice about which cluster to wander into)
    ("coffee", "kitchen", 2.0),      # food <-> work bridge (already above)
    ("park", "office", 0.5),          # weak outdoors <-> work bridge
    ("dog", "park", 1.0),             # pets <-> outdoors bridge (besides leash->walk)
    ("meeting", "kitchen", 0.5),      # weak work <-> food bridge
]
for a, b, w in edges:
    g.add_edge(a, b, w)

print("=== Graph summary ===")
g.describe()
print(f"\n{len(nodes)} nodes total across 4 loose clusters "
      f"(pets, outdoors, food, work) with 4 cross-links.")

# ---------- retrieval mode from a hub node ----------

print("\n=== Retrieval-mode walks from 'dog' (low randomness) ===")
for i in range(5):
    path = random_walk(g, "dog", energy=3.0, decay=0.6, randomness=0.3, rng=rng)
    print(f"  walk {i+1}: {' -> '.join(path)}")

# ---------- creative mode from the same hub ----------

print("\n=== Creative-mode walks from 'dog' (high randomness) ===")
for i in range(5):
    path = random_walk(g, "dog", energy=5.0, decay=0.75, randomness=2.0, rng=rng)
    print(f"  walk {i+1}: {' -> '.join(path)}")

# ---------- does reinforcing a cross-cluster pair open up new paths? ----------

print("\n=== Reinforcing 'dog' + 'coffee' together over 8 cycles (unrelated clusters) ===")
for _ in range(8):
    g.reinforce(["dog", "coffee"])
    g.step()

print("\nNew dog--coffee weight:", g.neighbors("dog").get("coffee"))
print("\n=== Creative-mode walks from 'dog' AFTER reinforcement ===")
for i in range(5):
    path = random_walk(g, "dog", energy=5.0, decay=0.75, randomness=2.0, rng=rng)
    print(f"  walk {i+1}: {' -> '.join(path)}")

# ---------- variety check: how many distinct paths across many walks? ----------

print("\n=== Variety check: 20 creative walks from 'dog', counting distinct paths ===")
paths = multi_walk(g, "dog", n_walks=20, energy=4.0, decay=0.7, randomness=1.8, rng=rng)
distinct = {tuple(p) for p in paths}
print(f"  {len(distinct)} distinct paths out of 20 walks")
for p in list(distinct)[:8]:
    print(f"    {' -> '.join(p)}")