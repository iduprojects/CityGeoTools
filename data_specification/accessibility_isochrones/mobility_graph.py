import graph_tool

def validate_graph(G):

    edge_validity = {}
    node_validity = {}
    edge_properties = G.ep
    node_properties = G.vp

    edge_validity["length_meter"] = True if "length_meter" in edge_properties else False
    edge_validity["time_min"] = True if "time_min" in edge_properties else False

    node_validity["x"] = True if "x" in node_properties else False
    node_validity["y"] = True if "y" in node_properties else False
    node_validity["stop"] = True if "stop" in node_properties else False

    return node_validity, edge_validity