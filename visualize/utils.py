from pyvis import network

def visualize(graph, path, notebook=False, heading=""):
    g = network.Network(directed=True,notebook=notebook, heading=heading, select_menu=True, filter_menu=True)
    g.from_nx(graph)
    g.show(path)