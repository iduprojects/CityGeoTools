import osmnx as ox
import networkx as nx
import momepy
import geopandas as gpd
import shapely
import numpy as np
import pandas as pd
import osm2geojson
import math

from utils.overpass_query import *
from utils import transform
from shapely.geometry import LineString
from tqdm import tqdm
tqdm.pandas()

# graph type must be 'walk' or 'drive'
def get_osmnx_graph(city_osm_id, city_crs, graph_type, speed=None):

    boundary = overpass_get_boundary(city_osm_id)  
    boundary = osm2geojson.json2geojson(boundary)
    boundary = gpd.GeoDataFrame.from_features(boundary["features"]).set_crs(4326)

    print(f"Extracting and preparing {graph_type} graph...")
    G_ox = ox.graph.graph_from_polygon(polygon=boundary["geometry"][0], network_type=graph_type)
    G_ox.graph["approach"] = "primal"

    nodes, edges = momepy.nx_to_gdf(G_ox, points=True, lines=True, spatial_weights=False)
    nodes = nodes.to_crs(city_crs).set_index("nodeID")
    nodes_coord = nodes.geometry.apply(
        lambda p: {"x": round(p.coords[0][0], 2), "y": round(p.coords[0][1], 2)}
        ).to_dict()

    print(edges)
    edges = edges[["length", "node_start", "node_end", "geometry"]].to_crs(city_crs)
    edges["type"] = graph_type
    edges["geometry"] = edges["geometry"].apply(
        lambda x: LineString([tuple(round(c, 2) for c in n) for n in x.coords])
        )

    travel_type = "walk" if graph_type == "walk" else "car"
    if not travel_speed:
        travel_speed =  4 * 1000 / 60 if graph_type == "walk" else  17 * 1000 / 60

    G = nx.MultiDiGraph()
    for i, edge in tqdm(edges.iterrows(), total=len(edges)):
        p1 = int(edge.node_start)
        p2 = int(edge.node_end)
        geometry = shapely.geometry.LineString(
            ([(nodes_coord[p1]["x"], nodes_coord[p1]["y"]), (nodes_coord[p2]["x"], nodes_coord[p2]["y"])])
            ) if not edge.geometry else edge.geometry
        G.add_edge(
            p1, p2, length_meter=edge.length, geometry=str(geometry), type = travel_type, 
            time_min = math.ceil(edge.length / travel_speed)
            )
    nx.set_node_attributes(G, nodes_coord)
    G.graph['crs'] = 'epsg:' + str(city_crs)
    G.graph['graph_type'] = travel_type + " graph"
    G.graph['speeds'] = {travel_type: travel_speed}

    print(f"{graph_type.capitalize()} graph done!")
    return G


def public_routes_to_edges(city_osm_id, city_crs, transport_type, speed):

    routes = overpass_get_routes(city_osm_id, transport_type)
    print(f"Extracting and preparing {transport_type} routes:")
    df_routes = routes.progress_apply(
        lambda x: parse_overpass_route_response(x, city_crs), axis = 1, result_type="expand"
        )
    df_routes = gpd.GeoDataFrame(df_routes).dropna(subset=["way"]).set_geometry("way")

    # some stops don't lie on lines, therefore it's needed to project them
    stop_points = df_routes.apply(lambda x: transform.project_platforms(x, city_crs), axis = 1)

    edges = []
    time_on_stop = 1
    for i, route in stop_points.iterrows():
        length = np.diff(list(route["distance"]))
        for j in range(len(route["pathes"])):
            p1 = route["pathes"][j][0]
            p2 = route["pathes"][j][1]
            d = {"time": math.ceil(length[j]/speed + time_on_stop), "length": round(length[j], 2), 
                "type": transport_type, "desc": f"route {i}", "geometry": str(LineString([p1, p2]))}
            edges.append((p1, p2, d))

    return edges


def get_public_trasport_graph(city_osm_id, city_crs, transport_types_speed):

    G = nx.MultiDiGraph()
    edegs_different_types = []

    for transport_type, speed in transport_types_speed.items():
        edges = public_routes_to_edges(city_osm_id, city_crs, transport_type, speed)
        edegs_different_types.append(edges)
    
    G.add_edges_from(edegs_different_types)
    node_attributes = {node: {
        "x": round(node[0], 2), "y": round(node[1], 2), "stops": True, "desc": [] 
        } for node in list(G.nodes)}
        
    for p1, p2, data in list(G.edges(data=True)):
        transport_type = data["type"]
        node_attributes[p1]["desc"].append(transport_type), node_attributes[p2]["desc"].append(transport_type)

    for data in node_attributes.values():
        data["desc"] = ", ".join(set(data["desc"]))
    nx.set_node_attributes(G, node_attributes)
    G = nx.convert_node_labels_to_integers(G)
    G.graph['crs'] = 'epsg:' + str(city_crs)
    G.graph['graph_type'] = "public transport graph"
    G.graph['speeds'] = transport_types_speed


    print("Public transport graph done!")
    return G


# G_base - the biggest one 
def graphs_spatial_union(G_base, G_to_project):
    
    points = gpd.GeoDataFame([[n, Point((d["x"], d["y"]))] for n, d in G_to_project.nodes(data=True)],
                            columns=["node_id_to_project", "geometry"])
    edges_geom = transform.get_nearest_edge_geometry(points, G_base)
    projected_point_info = transform.project_point_on_edge(edges_geom)
    check_point_on_line = projected_point_info.apply(
        lambda x: x.edge_geometry.buffer(1).contains(x.nearest_point_geometry), axis=1).all()
    if not check_point_on_line:
        raise ValueError("Some projected points don't lie on edges")
        
    updated_G_base, points_df = transform.update_edges(projected_point_info, G_base)
    united_graph = transform.join_graph(updated_G_base, G_to_project, points_df)

    return united_graph


def get_intermodal_graph(city_osm_id, city_crs, public_transport_speeds, 
                        drive_speed =  4 * 1000 / 60, walk_speed = 17 * 1000 / 60):

    walk_graph = get_osmnx_graph(city_osm_id, city_crs, "walk", speed=walk_speed)
    drive_graph = get_osmnx_graph(city_osm_id, city_crs, "drive", speed=drive_speed)
    public_transport_graph = get_public_trasport_graph(city_osm_id, city_crs, public_transport_speeds)

    intermodal_graph = graphs_spatial_union(walk_graph, drive_graph)
    intermodal_graph = graphs_spatial_union(intermodal_graph, public_transport_graph)

    return intermodal_graph