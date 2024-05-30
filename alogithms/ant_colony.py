import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from graph_operations import generate_weighted_graph_matrix, save_graph_picture

@dataclass
class Ant:
    path: list[int] = field(default_factory=list)
    pheromone: list[int] = field(default_factory=list)
    target: int = -1
    is_reached_target: bool = False
    is_deadlock: bool = False
    path_length: int = 0

    def make_choice(self, G, alpha: float, beta: float) -> None:
        if self.is_deadlock or self.is_reached_target:
            return
        
        if self.target in G[self.path[-1]].keys():
            self.path.append(self.target)
            self.pheromone.append(1 / G[self.path[-2]][self.path[-1]]['weight'])
            self.path_length += G[self.path[-2]][self.path[-1]]['weight']
            self.is_reached_target = True
            return
        
        vertex_probabilities: dict = {}
        for vertex, info in  G[self.path[-1]].items():
            if vertex not in self.path:
                vertex_probabilities[vertex] =  (1 / info['weight'] ** beta) * info['pheromone'] ** alpha

        if not vertex_probabilities:
            self.is_deadlock = True
            return

        # norm probabilities
        prob_sum = sum(vertex_probabilities.values())
        for vertex, prob in vertex_probabilities.items():
            vertex_probabilities[vertex] = prob / prob_sum
        
        self.path.append(np.random.choice(list(vertex_probabilities.keys()), p=list(vertex_probabilities.values())))
        self.path_length += G[self.path[-2]][self.path[-1]]['weight']
        self.pheromone.append(1 / G[self.path[-2]][self.path[-1]]['weight'])


class AntColony:
    def __init__(self, n_ants: int, alpha: float, beta: float, evaporation: float) -> None:
        self._n_ants = n_ants
        self._alpha = alpha  # pheromone coefficient
        self._beta = beta  # distance coefficient
        self._evaporation = evaporation

        self._pheromone_init_value = 0.1


    def _init_pheromone(self, G) -> None:
        for u, v, d in G.edges(data=True):
            d['pheromone'] = self._pheromone_init_value


    def _init_ants(self, start_vertex: int, end_vertex: int) -> list[Ant]:
        ants: list[Ant] = [Ant() for _ in range(self._n_ants)]
        for i, a in enumerate(ants):
            a.path.append(start_vertex)
            a.target = end_vertex
        return ants

    def _update_pheromone(self, G, ants: list[Ant]) -> None:
        for u, v, d in G.edges(data=True):
            d['pheromone'] *= (1 - self._evaporation)

        for a in ants:
            for i in range(len(a.pheromone)):
                G[a.path[i]][a.path[i+1]]['pheromone'] += a.pheromone[i]


    
    def optimize(self, G, start_vertex: int, end_vertex: int, iterations: int) -> tuple[list[int], int] | None:
        G = G.copy()
        self._init_pheromone(G)
        best_ants: list[Ant] = []
        
        for _ in range(iterations):
            ants: list[Ant] = self._init_ants(start_vertex, end_vertex)
            reached_target_ants: list[Ant] = []

            while ants:
                for i, a in enumerate(ants):
                    if a.is_deadlock:
                        ants.pop(i)
                    if a.is_reached_target:
                        reached_target_ants.append(ants.pop(i))
                    a.make_choice(G, self._alpha, self._beta)

            self._update_pheromone(G, reached_target_ants)
            try:
                best_ants.append(max(reached_target_ants, key=lambda a: a.path_length))
            except ValueError:
                # All ants died
                continue

        try:
            res = max(best_ants, key=lambda a: a.path_length)
            return res.path, res.path_length
        except ValueError:
            # All ants died
            return None
        


if __name__ == '__main__':
    V_NUM: int = 4  # number of vertices(вершины)
    E_NUM: int = 5  # number of edges(ребра) 
    MIN_WEIGHT: int = 1
    MAX_WEIGHT: int = 100
    IS_ORIENTED: bool = False
    RANDOM_STATE: int = 44

    adjacency_matrix: np.ndarray = generate_weighted_graph_matrix(V_NUM, E_NUM, 
                                                                  MIN_WEIGHT, MAX_WEIGHT, 
                                                                  is_oriented=IS_ORIENTED, random_state=RANDOM_STATE)
    

    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph if IS_ORIENTED else nx.Graph)
    G.add_node(4)
    G.add_edge(2, 4, weight=5.0)
    save_graph_picture(G)

    ac = AntColony(10, 1, 1, 0.2)
    print(ac.optimize(G, 2, 1, 10))

