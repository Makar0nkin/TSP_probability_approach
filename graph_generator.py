import numpy as np
from itertools import product
from typing import Callable

def generate_weighted_graph_matrix(
        v_num: int, 
        e_num: int, 
        min_wight: int, 
        max_wight: int, 
        *, 
        is_oriented: bool = True,
        random_state: int | None = None
    ) -> np.ndarray:
    if is_oriented and e_num > v_num * (v_num - 1):
        raise ValueError("Too much edges")
    elif not is_oriented and e_num > v_num * (v_num - 1) / 2:
        raise ValueError("Too much edges")
    
    np.random.seed(random_state)

    adjacency_matrix: np.ndarray = np.zeros((v_num, v_num))
    weights: list[int] = np.floor(np.random.uniform(min_wight, max_wight, e_num)).tolist()
    key: Callable = (lambda c: c[0] != c[1]) if is_oriented else (lambda c: c[0] > c[1])
    possible_edges_coords: list[tuple[int, int]] = list(filter(key, product(range(v_num), repeat=2)))
    
    edges: np.ndarray = np.array(possible_edges_coords)[np.random.choice(len(possible_edges_coords), e_num, replace=False)]
    for (i, j) in edges:
        adjacency_matrix[i][j] = weights.pop()
        if not is_oriented:
            adjacency_matrix[j][i] = adjacency_matrix[i][j]
    return adjacency_matrix

if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    V_NUM: int = 4  # number of vertices(вершины)
    E_NUM: int = 5  # number of edges(ребра) 
    MIN_WEIGHT: int = 1
    MAX_WEIGHT: int = 10
    IS_ORIENTED: bool = False
    RANDOM_STATE: int = 44

    adjacency_matrix: np.ndarray = generate_weighted_graph_matrix(V_NUM, E_NUM, 
                                                                  MIN_WEIGHT, MAX_WEIGHT, 
                                                                  is_oriented=IS_ORIENTED, random_state=RANDOM_STATE)
    print(adjacency_matrix)

    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph if IS_ORIENTED else nx.Graph)
    pos=nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
