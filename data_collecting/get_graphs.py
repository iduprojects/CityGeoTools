from unicodedata import name
import osmnx as ox
import networkx as nx
import momepy
import geopandas as gpd
import shapely
import numpy as np
import pandas as pd
import osm2geojson

from data_collecting.utils.overpass_queries import *
from data_collecting.utils import transform
from shapely.geometry import LineString
from tqdm import tqdm
tqdm.pandas()

# graph type must be 'walk' or 'drive'
def get_osmnx_graph(city_osm_id, city_crs, graph_type, speed=None):

    boundary = overpass_query(get_boundary, city_osm_id)  
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

    edges = edges[["length", "node_start", "node_end", "geometry"]].to_crs(city_crs)
    edges["type"] = graph_type
    edges["geometry"] = edges["geometry"].apply(
        lambda x: LineString([tuple(round(c, 2) for c in n) for n in x.coords] if x else None)
        )

    travel_type = "walk" if graph_type == "walk" else "car"
    if not speed:
        speed =  4 * 1000 / 60 if graph_type == "walk" else  17 * 1000 / 60

    G = nx.MultiDiGraph()
    for i, edge in tqdm(edges.iterrows(), total=len(edges)):
        p1 = int(edge.node_start)
        p2 = int(edge.node_end)
        geometry = shapely.geometry.LineString(
            ([(nodes_coord[p1]["x"], nodes_coord[p1]["y"]), (nodes_coord[p2]["x"], nodes_coord[p2]["y"])])
            ) if not edge.geometry else edge.geometry
        G.add_edge(
            p1, p2, length_meter=edge.length, geometry=str(geometry), type = travel_type, 
            time_min = round(edge.length / speed)
            )
    nx.set_node_attributes(G, nodes_coord)
    G.graph['crs'] = 'epsg:' + str(city_crs)
    G.graph['graph_type'] = travel_type + " graph"
    G.graph[travel_type + ' speed'] = round(speed, 2)

    print(f"{graph_type.capitalize()} graph done!")
    return G


def public_routes_to_edges(city_osm_id, city_crs, transport_type, speed):

    routes = overpass_query(get_routes, city_osm_id, transport_type)
    print(f"Extracting and preparing {transport_type} routes:")

    try:
        df_routes = routes.progress_apply(
            lambda x: parse_overpass_route_response(x, city_crs), axis = 1, result_type="expand"
            )
        df_routes = gpd.GeoDataFrame(df_routes).dropna(subset=["way"]).set_geometry("way")

    except KeyError:
        print(f"It seems there are no {transport_type} routes in the city. This transport type will be skipped.")
        return []

    # some stops don't lie on lines, therefore it's needed to project them
    stop_points = df_routes.apply(lambda x: transform.project_platforms(x, city_crs), axis = 1)

    edges = []
    time_on_stop = 1
    for i, route in stop_points.iterrows():
        length = np.diff(list(route["distance"]))
        for j in range(len(route["pathes"])):
            p1 = route["pathes"][j][0]
            p2 = route["pathes"][j][1]
            d = {"time_min": round(length[j]/speed + time_on_stop), "length_meter": round(length[j], 2), 
                "type": transport_type, "desc": f"route {i}", "geometry": str(LineString([p1, p2]))}
            edges.append((p1, p2, d))

    return edges


def get_public_trasport_graph(city_osm_id, city_crs, transport_types_speed):

    G = nx.MultiDiGraph()
    edegs_different_types = []

    for transport_type, speed in transport_types_speed.items():
        edges = public_routes_to_edges(city_osm_id, city_crs, transport_type, speed)
        edegs_different_types.extend(edges)
    
    G.add_edges_from(edegs_different_types)
    node_attributes = {node: {
        "x": round(node[0], 2), "y": round(node[1], 2), "stop": True, "desc": [] 
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
    G.graph.update({k + " speed": round(v, 2) for k, v in transport_types_speed.items()})


    print("Public transport graph done!")
    return G


# G_base - the biggest one 
def graphs_spatial_union(G_base, G_to_project):
    
    points = gpd.GeoDataFrame([[n, Point((d["x"], d["y"]))] for n, d in G_to_project.nodes(data=True)],
                            columns=["node_id_to_project", "geometry"])
    edges_geom = transform.get_nearest_edge_geometry(points, G_base)
    projected_point_info = transform.project_point_on_edge(edges_geom)

    check_point_on_line = projected_point_info.apply(
        lambda x: x.edge_geometry.buffer(1).contains(x.nearest_point_geometry), axis=1).all()
    if not check_point_on_line:
        raise ValueError("Some projected points don't lie on edges")

    points_on_lines = projected_point_info[(projected_point_info["len_from_start"] != 0) 
                                            & (projected_point_info["len_to_end"] != 0)]
    points_on_points = projected_point_info[~projected_point_info.index.isin(points_on_lines.index)]
    points_on_points["connecting_node_id"] = points_on_points.apply(
        lambda x: x.edge_id[0] if x.len_from_start == 0 else x.edge_id[1], axis=1
    )
    
    updated_G_base, points_on_lines = transform.update_edges(points_on_lines, G_base)

    points_df = pd.concat([points_on_lines, points_on_points])
    united_graph = transform.join_graph(updated_G_base, G_to_project, points_df)

    return united_graph


def get_intermodal_graph(city_osm_id, city_crs, public_transport_speeds, 
                        drive_speed =  4 * 1000 / 60, walk_speed = 17 * 1000 / 60):

    G_walk = get_osmnx_graph(city_osm_id, city_crs, "walk", speed=walk_speed)
    nx.write_graphml(G_walk, "/var/essdata/IDU/other/mm_22/walk_graph.graphml")
    G_drive = get_osmnx_graph(city_osm_id, city_crs, "drive", speed=drive_speed)
    nx.write_graphml(G_drive, "/var/essdata/IDU/other/mm_22/drive_graph.graphml")
    G_public_transport = get_public_trasport_graph(city_osm_id, city_crs, public_transport_speeds)
    nx.write_graphml(G_public_transport, "/var/essdata/IDU/other/mm_22/public_transport_graph.graphml")

    print("Union of graphs...")
    G_intermodal = graphs_spatial_union(G_walk, G_drive)
    G_intermodal = graphs_spatial_union(G_intermodal, G_public_transport)

    for u, v, d in G_intermodal.edges(data=True):
        if "time_min" not in d:
            d["time_min"] = round(d["length_meter"] / walk_speed)
        if "desc" not in d:
            d["desc"] = ""

    for u, d in G_intermodal.nodes(data=True):
        if "stop" not in d:
            d["stop"] = False
        if "desc" not in d:
            d["desc"] = ""

    G_intermodal.graph["graph_type"] = "intermodal graph"
    G_intermodal.graph["drive speed"] = drive_speed
    G_intermodal.graph.update({k: v for k, v in G_public_transport.graph.items() if "speed" in k})
    G_intermodal.graph["created by"] = "CityGeoTools"

    print("Intermodal graph done!")
    return G_intermodal