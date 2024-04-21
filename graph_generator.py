import random
import math


def random_pick(n: int, m: int) -> list[int]:
    """Pick m integers from a bag of the integers in [0, n) without replacement"""
    d: dict[int, int] = {i : i for i in range(m)} # For now, just pick the first m integers
    res: list[int] = []
    for i in range(m): # Pick the i-th number
        j = random.randrange(i, n)
        # Pick whatever is in the j-th slot. If there is nothing, then pick j.
        if j not in d:
            d[j] = j
        d[i], d[j] = d[j], d[i] # Swap the contents of the i-th and j-th slot
        res.append(d[i])
    return res

def gen_random_graph(V, E):
    """Generate an undirected graph in the form of an adjacency list with no duplicate edges or self loops"""
    g = [[] for _ in range(V)]
    edges = random_pick(math.comb(V, 2), E) # Pick E integers that represent the edges
    for e in edges: # Decode the edges into their vertices
        u = int((1 + math.sqrt(1 + 8 * e)) / 2)
        v = e - math.comb(u, 2)
        g[u].append(v)
        g[v].append(u)
    return g


if __name__ == "__main__":
    import networkx as nx

    # The complete graph on 4 vertices
    adjacency_matrix: list[list[int]] = gen_random_graph(4, 6)
    G = nx.from_numpy_array(adjacency_matrix)

    print(adjacency_matrix)
    nx.draw(G)