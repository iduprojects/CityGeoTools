from json import loads
import json
import pyproj
import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely

def request_points_project(request_points, set_crs, to_crs):
        transformer = pyproj.Transformer.from_crs(set_crs, to_crs)
        return [transformer.transform(point[0], point[1]) for point in request_points]

def geojson_projection_management(geojson, set_crs, to_crs):
    gdf = gpd.GeoDataFrame.from_features(geojson['features'])
    gdf = gdf.set_crs(set_crs).to_crs(to_crs)
    return json.loads(gdf.to_json())

def routes_between_two_points(graph, p1: tuple, p2: tuple, weight:str):

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
    
    

