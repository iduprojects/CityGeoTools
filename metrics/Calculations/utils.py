from json import loads
import json
import pyproj
import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely
import networkit as nk

def request_points_project(request_points, set_crs, to_crs):
        transformer = pyproj.Transformer.from_crs(set_crs, to_crs)
        return [transformer.transform(point[0], point[1]) for point in request_points]

def geojson_projection_management(geojson, set_crs, to_crs):
    gdf = gpd.GeoDataFrame.from_features(geojson['features'])
    gdf = gdf.set_crs(set_crs).to_crs(to_crs)
    return json.loads(gdf.to_json())

def routes_between_two_points(graph, p1: tuple, p2: tuple, weight=None):

    try:
        graph_path = nx.dijkstra_path(graph, weight=weight, 
                                      source=ox.distance.nearest_nodes(graph, *p1),
                                      target=ox.distance.nearest_nodes(graph, *p2))
    except nx.NetworkXNoPath:
        return None

    complete_route = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in graph_path]
    route_geometry = shapely.geometry.LineString(complete_route)
    route_len = nx.path_weight(graph, graph_path, weight)
    
    return {"route_geometry": route_geometry, "route_len": route_len}


def convert_nx2nk(G_nx, weight=None):

    idmap = dict((id, u) for (id, u) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
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
        for u_, v_ in tqdm(edges):
                u, v = idmap[u_], idmap[v_]
                G_nk.addEdge(u, v)
    
    

