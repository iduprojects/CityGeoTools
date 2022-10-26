import json
import pyproj
import geopandas as gpd
import networkx as nx
import shapely
import pandas as pd
import networkit as nk

from scipy import spatial


def request_points_project(request_points, set_crs, to_crs):
        transformer = pyproj.Transformer.from_crs(set_crs, to_crs)
        return [transformer.transform(point[0], point[1]) for point in request_points]


def geojson_projection_management(geojson, set_crs, to_crs):
    gdf = gpd.GeoDataFrame.from_features(geojson['features'])
    gdf = gdf.set_crs(set_crs).to_crs(to_crs)
    return json.loads(gdf.to_json())

def nk_routes_between_two_points(G_nk, nodes_data, p1, p2, exact_geometry=False):
    
    distance, p_ = spatial.cKDTree(nodes_data).query([p1, p2])
    p1_ = nodes_data.iloc[p_[0]].name
    p2_ = nodes_data.iloc[p_[1]].name
    dijkstra = nk.distance.Dijkstra(G_nk, source=p1_, target=p2_, storePaths=True)
    dijkstra.run()
    route_len = round(dijkstra.distance(p2_) + distance.sum(), 2)

    if exact_geometry:
        complete_route = [(nodes_data[n]["x"], nodes_data[n]["y"]) for n in dijkstra.getPath(p2_)]
        route_geometry = shapely.geometry.LineString(complete_route)
    else:
        route_geometry =  shapely.geometry.LineString([p1, p2])
    return {"route_geometry": route_geometry, "route_len": route_len}
    

def nx_routes_between_two_points(G_nx, p1, p2, weight, exact_geometry=False):

    nodes_data = pd.DataFrame.from_records(
            [{"x": d["x"], "y": d["y"]} for u, d in G_nx.nodes(data=True)], index=list(G_nx.nodes())
            ).sort_index()
    distance, p_ = spatial.cKDTree(nodes_data).query([p1, p2])
    p1_ = nodes_data.iloc[p_[0]].name
    p2_ = nodes_data.iloc[p_[1]].name
    try:
        graph_path = nx.dijkstra_path(G_nx, weight=weight, source=p1_, target=p2_)
        route_len = round(nx.path_weight(G_nx, graph_path, weight) + distance.sum(), 2)
    except nx.NetworkXNoPath:
        return None

    if exact_geometry:
        complete_route = [(G_nx.nodes[n]['x'], G_nx.nodes[n]['y']) for n in graph_path]
        route_geometry = shapely.geometry.LineString(complete_route)
    else:
        route_geometry =  shapely.geometry.LineString([p1, p2])
    return {"route_geometry": route_geometry, "route_len": route_len}