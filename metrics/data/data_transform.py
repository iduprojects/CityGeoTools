import shapely
import networkit as nk


def load_graph_geometry(graph, node=True, edge=False):

    if edge:
        for u, v, data in graph.edges(data=True):
            data["geometry"] = shapely.wkt.loads(data["geometry"])
    if node:
        for u, data in graph.nodes(data=True):
            data["geometry"] = shapely.geometry.Point([data["x"], data["y"]])

    return graph

def get_nx2_nk_idmap(G_nx):
    idmap = dict((id, u) for (id, u) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
    return idmap

def get_nk_attrs(G_nx):
    attrs = dict(
        (u, {"x": d[-1]["x"], "y": d[-1]["y"]}) 
        for (d, u) in zip(G_nx.nodes(data=True), range(G_nx.number_of_nodes()))
        )
    return attrs

def convert_nx2nk(G_nx, idmap=None, weight=None):

    if not idmap:
        idmap = get_nx2_nk_idmap(G_nx)
    n = max(idmap.values()) + 1
    edges = list(G_nx.edges())

    if weight:
        G_nk = nk.Graph(n, directed=G_nx.is_directed(), weighted=True)
        for u_, v_ in edges:
                u, v = idmap[u_], idmap[v_]
                d = dict(G_nx[u_][v_])
                if len(d) > 1:
                    for d_ in d.values():
                            v__ = G_nk.addNodes(2)
                            u__ = v__ - 1
                            w = d_[weight] if weight in d_ else 1
                            G_nk.addEdge(u, v, w)
                            G_nk.addEdge(u_, u__, 0)
                            G_nk.addEdge(v_, v__, 0)
                else:
                    d_ = list(d.values())[0]
                    w = d_[weight] if weight in d_ else 1
                    G_nk.addEdge(u, v, w)
    else:
        G_nk = nk.Graph(n, directed=G_nx.is_directed())
        for u_, v_ in edges:
                u, v = idmap[u_], idmap[v_]
                G_nk.addEdge(u, v)

    return G_nk