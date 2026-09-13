"""
Retrieval via weighted random walk.

At each step, from the current node, the next node is sampled from its
neighbors with probability proportional to edge weight. The walk carries
a decaying "energy" budget so it naturally terminates.

randomness (temperature) controls how much weight differences matter:
  - low randomness  -> walk almost always follows the thickest edge
                        ("retrieval" mode: strict, predictable)
  - high randomness -> walk samples more uniformly across edges
                        ("creative" mode: looser, more surprising)
"""

import random


def _weighted_choice(neighbors: dict[str, float], randomness: float, rng: random.Random):
    """
    neighbors: {node_id: weight}
    randomness: temperature in (0, inf). 0 -> always pick the max weight
                (approximated here with a very small temperature floor);
                1.0 -> sample proportional to raw weights;
                >1.0 -> flatter, more uniform sampling.
    """
    if not neighbors:
        return None

    ids = list(neighbors.keys())
    weights = list(neighbors.values())

    temp = max(randomness, 1e-6)
    # softmax-style reweighting: raise weights to power 1/temp, low temp
    # sharpens toward the max, high temp flattens toward uniform.
    adjusted = [w ** (1.0 / temp) if w > 0 else 0.0 for w in weights]
    total = sum(adjusted)
    if total <= 0:
        return rng.choice(ids)

    r = rng.uniform(0, total)
    upto = 0.0
    for node_id, w in zip(ids, adjusted):
        upto += w
        if upto >= r:
            return node_id
    return ids[-1]  # floating point fallback


def random_walk(graph, start_id: str, energy: float = 3.0, decay: float = 0.6,
                 randomness: float = 1.0, min_energy: float = 0.3,
                 avoid_immediate_backtrack: bool = True, rng: random.Random | None = None):
    """
    Runs a single weighted random walk starting at start_id.

    energy:      starting budget. The walk stops once energy drops below
                 min_energy or there are no unvisited-enough neighbors.
    decay:       multiplicative energy loss per hop (e.g. 0.6 means energy
                 *= 0.6 after every hop -> fewer, closer hops).
    randomness:  passed through to the sampling temperature (see
                 _weighted_choice).
    Returns: list of node_ids visited, in order (including start_id).
    """
    rng = rng or random.Random()
    path = [start_id]
    current = start_id
    previous = None

    while energy >= min_energy:
        neighbors = dict(graph.neighbors(current))
        if avoid_immediate_backtrack and previous is not None:
            neighbors.pop(previous, None)
        if not neighbors:
            break

        nxt = _weighted_choice(neighbors, randomness, rng)
        if nxt is None:
            break

        path.append(nxt)
        previous, current = current, nxt
        energy *= decay

    return path


def multi_walk(graph, start_id: str, n_walks: int = 3, **walk_kwargs):
    """
    Runs several independent walks from the same start node — e.g. to get
    several distinct trains of association per query. Returns a list of
    paths (one per walk).
    """
    rng = walk_kwargs.pop("rng", None) or random.Random()
    return [random_walk(graph, start_id, rng=rng, **walk_kwargs) for _ in range(n_walks)]
