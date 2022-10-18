from pyvis import network
import networkx as nx
from sklearn import neighbors

def visualize(graph: nx.Graph, path: str, notebook: bool=False, heading: str=""):
    g = network.Network(directed=True,notebook=notebook, heading=heading, select_menu=True, filter_menu=True)
    g.from_nx(graph)
    g.show(path)

def add_neighbours(subgraph: nx.Graph, graph: nx.Graph):
    "The subgraph called here should not have modified attributes"
    neighbors = []
    _graph = graph.to_undirected()
    for n in subgraph:
        neighbors.extend(_graph.neighbors(n))
    
    neighbors = set(neighbors)
    subgraph_nodes = set(subgraph)

    new_subgraph = graph.subgraph(neighbors.union(subgraph_nodes)).copy()

    for n in neighbors.difference(subgraph_nodes):
        new_subgraph.nodes[n]['color'] = "#808080"

    return new_subgraph