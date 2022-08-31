def validate_graph(G):

    edge_validity = {}
    node_validity = {}

    types = set([e[-1]["type"] for e in G.edges(data=True)])
    edge_validity["type"] = len(types) > 0
    edge_validity["walk value in type"] = 'walk' in types
    edge_validity["length_meter"] = all(["length_meter" in e[-1] for e in G.edges(data=True)])
    edge_validity["time_min"] = all(["time_min" in e[-1] for e in G.edges(data=True)])

    node_validity["x"] = all(["x" in n[-1] for n in G.nodes(data=True)])
    node_validity["y"] = all(["y" in n[-1] for n in G.nodes(data=True)])

    return node_validity, edge_validity