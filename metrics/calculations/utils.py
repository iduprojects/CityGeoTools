import json
import pyproj
import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely
import pandas as pd

from scipy import spatial


def request_points_project(request_points, set_crs, to_crs):
        transformer = pyproj.Transformer.from_crs(set_crs, to_crs)
        return [transformer.transform(point[0], point[1]) for point in request_points]


def geojson_projection_management(geojson, set_crs, to_crs):
    gdf = gpd.GeoDataFrame.from_features(geojson['features'])
    gdf = gdf.set_crs(set_crs).to_crs(to_crs)
    return json.loads(gdf.to_json())


def routes_between_two_points(graph, p1: tuple, p2: tuple, weight, exact_geometry=False):

    nodes_data = pd.DataFrame.from_records(
            [{"x": d["x"], "y": d["y"]} for u, d in graph.nodes(data=True)], index=list(graph.nodes())
            ).sort_index()
    distance, p_ = spatial.KDTree(nodes_data).query([p1, p2])
    p1_ = nodes_data.iloc[p_[0]].name
    p2_ = nodes_data.iloc[p_[1]].name
    try:
        graph_path = nx.dijkstra_path(graph, weight=weight, source=p1_, target=p2_)
        route_len = round(nx.path_weight(graph, graph_path, weight) + distance.sum(), 2)
    except nx.NetworkXNoPath:
        return None

    if exact_geometry:
        complete_route = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in graph_path]
        route_geometry = shapely.geometry.LineString(complete_route)
    else:
        route_geometry =  shapely.geometry.LineString([p1, p2])
    
    return {"route_geometry": route_geometry, "route_len": route_len}