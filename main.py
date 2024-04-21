import numpy as np
from alogithms import generate_weighted_graph_matrix, AntColony

if __name__ == '__main__':
    V_NUM: int = 4  # number of vertices(вершины)
    E_NUM: int = 5  # number of edges(ребра) 
    MIN_WEIGHT: int = 1
    MAX_WEIGHT: int = 10
    IS_ORIENTED: bool = False
    RANDOM_STATE: int = 44
    graph_matrix: np.ndarray = generate_weighted_graph_matrix(V_NUM, E_NUM, 
                                                              MIN_WEIGHT, MAX_WEIGHT, 
                                                              is_oriented=IS_ORIENTED, random_state=RANDOM_STATE)
    
    ant_colony = AntColony(graph_matrix, E_NUM, E_NUM, 100, 0.95, alpha=1, beta=1)
    shortest_path = ant_colony.run()
    print (f"shorted_path: {shortest_path}")