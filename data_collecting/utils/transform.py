import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx

from itertools import chain
from shapely.ops import nearest_points, substring
from shapely.geometry import LineString, Point



"""

Function get_linestring and its supplementary functions are used to fix topological errors
such as undershoot, overshoot and separate pieces of lines not related to main routes.

Function get_linestring takes a GeoDataFrame object with only one column 'geometry' where
each row contains a shapely MultiLineString object.

route: GeoDataFrame object

"""
def get_linestring(route):
    
    equal_lines = route.apply(lambda x: find_equals_line(x, route), axis=1).dropna()
    lines_todel = list(chain(*[line[1:] for line in list(equal_lines)]))
    route = route.drop(lines_todel).reset_index()
    
    path_buff = gpd.GeoSeries(route.geometry.buffer(0.01))
    connect_series = route.apply(lambda x: find_connection(x, path_buff), axis=1).dropna()
    sequences = get_sequences(connect_series, [])
    if sequences is None:
        return None

    len_sequence = [len(sec) for sec in sequences]
    max_sequence = len_sequence.index(max(len_sequence))
    sequence = sequences[max_sequence]
    comlete_line = [route.geometry[sequence[0]]]

    for i in range(len(sequence) - 1):
        line1 = comlete_line[i]
        line2 = route.geometry[sequence[i + 1]]
        con_point1, con_point2 = nearest_points(line1, line2)
        
        line2 = list(line2.coords)
        check_reverse = gpd.GeoSeries([Point(line2[0]), Point(line2[-1])]).distance(con_point2).idxmin()
        if check_reverse == 1:
            line2.reverse()
            
        comlete_line.append(LineString(line2))
    
    comlete_line = list(chain(*[list(line.coords) for line in comlete_line]))
    comlete_line = list(pd.Series(comlete_line).drop_duplicates())
    
    return LineString(comlete_line)


def find_equals_line(loc, series):
    
    series = series.drop(loc.name)
    eq_lines = series.geometry.apply(lambda x: x.almost_equals(loc.geometry))
    eq_lines = series[eq_lines].index
    
    equal_lines = sorted(list(eq_lines) + [loc.name]) if len(eq_lines) > 0 else None

    return equal_lines


def find_connection(loc, df):
    
    df = df.drop(loc.name)
    bool_ser = df.intersects(loc.geometry)
    connect_lines = df[bool_ser].index
    
    if len(connect_lines) > 0:
        return list(connect_lines)
    else:
        return None
        

def get_sequences(connect_ser, sequences=[]):
    
    num_con = connect_ser.apply(lambda x: len(x))
    finite_points = pd.DataFrame(connect_ser[num_con == 1].rename("value"))
    
    if len(finite_points) == 0:
        return None
    
    sequence = move_next(finite_points.index[0], connect_ser, [])
    sequences.append(sequence)
    
    route_finite = finite_points.index.isin(sequence)
    if route_finite.all():
        return sequences
    else:
        connect_ser = connect_ser.drop(finite_points.index[route_finite])
        sequences = get_sequences(connect_ser, sequences)
        return sequences

def move_next(path, series, sequence, branching=None):
    
    sequence.append(path)
    try:
        series = series.drop(path)
    except: pass
    bool_next_path = series.apply(lambda x: path in x)
    next_path = series[bool_next_path].index

    if len(next_path) == 0:
        return sequence
    
    elif len(next_path) > 1:
        
        if branching is None:
            branches_start = path
            sequence_variance = []
            for path in next_path:
                series_ = series.drop([path_ for path_ in next_path if path_ != path])
                sequence_ = move_next(path, series_, [], branches_start)
                sequence_variance.append(sequence_)
                
        else:
            return sequence
        
        len_sequence = [len(sec) for sec in sequence_variance]
        max_sequence = len_sequence.index(max(len_sequence))
        sequence_ = sequence_variance[max_sequence]
        series_ = series.drop(list(chain(*[sec[-2:] for sec in sequence_variance])))
        sequence = sequence + sequence_
        sequence_ = move_next(sequence_[-1], series_, sequence_, None)
        return sequence + sequence_
    
    else:
        sequence = move_next(next_path[0], series, sequence, branching)
    return sequence



"""

Function project_platforms and its supplementary functions are used to project points on lines.
It is a necessary operation since OpenStreetMap contains two types of points describing
public transport stops - 'stop' and 'platforms'. The points marked as 'platform' usually
do not lie on route lines. Moreover some of them are very close to each other and probably
mean the same stop. To check this, 'project_threshold' value and recursion function are used.

Function project_platforms takes two arguments - 'loc' which is Series contains rows 'platforms' and 'way'
and 'city_crs'. 'way' is shapely LineString object, and 'platforms' is Series of shapely Point objects

loc: Series object
city_crs: int


"""

def project_platforms(loc, city_crs):
    
    project_threshold = 5
    edge_indent = 10

    platforms = loc["platforms"]
    line = loc["way"]
    line_length = line.length

    if platforms is not None:
        platforms = gpd.GeoSeries(platforms).set_crs(4326).to_crs(city_crs)
        stops = platforms.apply(lambda x: nearest_points(line, x)[0])
        stops = gpd.GeoDataFrame(stops).rename(columns={0:"geometry"}).set_geometry("geometry")
        stops = recursion(stops, project_threshold)

        check_reverse = gpd.GeoSeries([Point(line.coords[0]), Point(line.coords[-1])]).distance(stops[0]).idxmin()
        if check_reverse == 1:
            line = list(line.coords)
            line.reverse()
            line = LineString(line)

        stops_distance = stops.apply(lambda x: line.project(x)).sort_values()
        stops = stops.loc[stops_distance.index]
        condition = (stops_distance > edge_indent)&(stops_distance < line_length - edge_indent)
        stops, distance = stops[condition].reset_index(drop=True), [0] + list(stops_distance[condition])
        distance.append(line_length)

        if len(stops) > 0:
            start_line = gpd.GeoSeries(Point(line.coords[0]))
            end_line = gpd.GeoSeries(Point(line.coords[-1]))
            stops = pd.concat([start_line, stops, end_line]).reset_index(drop=True)
        else:
            stops, distance = get_line_from_start_to_end(line, line_length)     
    else:
        stops, distance = get_line_from_start_to_end(line, line_length)

    pathes = [[tuple(round(c, 2) for c in stops[i].coords[0]), 
               tuple(round(c, 2) for c in stops[i + 1].coords[0])] 
               for i in range(len(stops) - 1)]

    return pd.Series({"pathes": pathes, "distance": distance})


def recursion (stops, threshold):
    
    stops['to_del'] = stops.apply(lambda x: get_index_to_delete(stops, x, threshold), axis = 1)
    
    if stops['to_del'].isna().all():
        return stops["geometry"]
    else:
        stops_near_pair = stops.dropna().apply(lambda x: sorted([x.name, x.to_del]), axis=1)
        stops_to_del = [pair[0] for pair in stops_near_pair]
        stops = stops.drop(stops_to_del)
        stops = recursion(stops, threshold)
    
    return stops.reset_index(drop=True)


def get_index_to_delete(other_stops, loc_stop, threshold):
    
    stops_to_del = other_stops.geometry.distance(loc_stop.geometry).sort_values().drop(loc_stop.name)
    stops_to_del = list(stops_to_del[stops_to_del < threshold].index)
    
    if len(stops_to_del) > 0:
        return stops_to_del[0]
    else:
        return None


def get_line_from_start_to_end(line, line_length):
    
    start_line = gpd.GeoSeries(Point(line.coords[0]))
    end_line = gpd.GeoSeries(Point(line.coords[-1]))
    stops = pd.concat([start_line, end_line]).reset_index(drop=True)
    distance = [0, line_length]
    
    return stops, distance

"""


"""

def get_nearest_edge_geometry(points, walk_graph):
    
    coords = list(points.geometry.apply(lambda x: list(x.coords)[0]))
    x = [c[0] for c in list(coords)]
    y = [c[1] for c in list(coords)]
    edges, distance = ox.distance.nearest_edges(walk_graph, x, y, return_dist=True)
    edges_geom = list(map(lambda x: (x, walk_graph[x[0]][x[1]][x[2]]["geometry"]), edges))
    edges_geom = pd.DataFrame(edges_geom, index=points.index, columns=["edge_id", "edge_geometry"])
    edges_geom["distance_to_edge"] = distance

    return pd.concat([points, edges_geom], axis=1)


def project_point_on_edge(points_edge_geom):
    
    points_edge_geom["nearest_point_geometry"] = points_edge_geom.apply(
        lambda x: nearest_points(x.edge_geometry, x.geometry)[0], axis=1)
    points_edge_geom["len"] = points_edge_geom.apply(
        lambda x: x.edge_geometry.length, axis=1)
    points_edge_geom["len_from_start"] = points_edge_geom.apply(
        lambda x: x.edge_geometry.project(x.geometry), axis=1)
    points_edge_geom["len_to_end"] = points_edge_geom.apply(
        lambda x: x.edge_geometry.length - x.len_from_start, axis=1)
    points_edge_geom = points_edge_geom[(points_edge_geom["len_from_start"] != 0) 
                                        & (points_edge_geom["len_to_end"] != 0)]
    return points_edge_geom


def update_edges(points_info, G):

    G_with_drop_edges = delete_edges(points_info, G)
    updated_G, split_points = add_splitted_edges(G_with_drop_edges, points_info)
    updated_G, split_points = add_connecting_edges(updated_G, split_points)

    return updated_G, split_points


def delete_edges(project_points, G):
    
    G_copy = G.copy()
    bunch_edges = []
    for e in list(project_points["edge_id"]):
        flag = check_parallel_edge(G_copy, *e)
        if flag == 2:
            bunch_edges.extend([(e[0], e[1], e[2]), (e[1], e[0], e[2])])
        else:
            bunch_edges.append((e[0], e[1], e[2]))
    
    bunch_edges = list(set(bunch_edges))
    G_copy.remove_edges_from(bunch_edges)
    
    return G_copy


def check_parallel_edge(G, u, v, n):
    if u == v:
        return 1
    elif G.has_edge(u, v) and G.has_edge(v, u):
        if G[u][v][n]["geometry"].equals(G[v][u][n]["geometry"]):
            return 2
        else:
            return 1
    else:
        return 1


def add_splitted_edges(G, split_nodes):
    
    start_node_idx = max((G.nodes)) + 1
    split_nodes["node_id"] = range(start_node_idx, start_node_idx + len(split_nodes))
    nodes_bunch = split_nodes.apply(lambda x: generate_nodes_bunch(x), axis=1)
    nodes_attr = split_nodes.set_index("node_id").nearest_point_geometry.apply(
        lambda x: {"x": list(x.coords)[0][0], "y": list(x.coords)[0][1]}).to_dict()
    G.add_edges_from(list(nodes_bunch.explode()))
    nx.set_node_attributes(G, nodes_attr)
    
    return G, split_nodes
        
    
def generate_nodes_bunch(split_point):
    
    edge_pair = []
    edge_nodes = split_point.edge_id
    edge_geom = split_point.edge_geometry
    new_node_id = split_point.node_id
    len_from_start = split_point.len_from_start
    len_to_end = split_point.len_to_end
    len_edge = split_point.len
    walk_speed = 5*1000/60
    
    fst_edge_attr = {'length': len_from_start, "geometry": substring(edge_geom, 0, len_from_start),
                     "time": len_from_start / walk_speed, "osm_id": "None", 'highway': "splitted"}
    snd_edge_attr = {'length': len_to_end, "geometry": substring(edge_geom, len_from_start, len_edge),
                     "time": len_to_end / walk_speed, "osm_id": "None", 'highway': "splitted"}
    edge_pair.extend([(edge_nodes[0], new_node_id, fst_edge_attr),(new_node_id, edge_nodes[0], fst_edge_attr),
                      (new_node_id, edge_nodes[1], snd_edge_attr), (edge_nodes[1], new_node_id, snd_edge_attr)])
    
    return edge_pair


def add_connecting_edges(G, split_nodes):

    start_node_idx = split_nodes["node_id"].max() + 1
    split_nodes["connecting_node_id"] = list(range(start_node_idx, start_node_idx + len(split_nodes)))
    connecting_edges = split_nodes.apply(
        lambda x: (x.node_id, x.connecting_node_id, {
            "type": "walk", "length_meter": x.distance_to_edge, 
            "geometry": str(LineString([x.geometry, x.nearest_point_geometry]))
            }), axis=1).tolist()
    G.add_edges_from(connecting_edges)
    return G, split_nodes


def join_graph(G_base, G_to_project, points_df):
    new_nodes = points_df.set_index("node_id_to_project")["connecting_node_id"]
    edge_to_add = [(new_nodes[n1], new_nodes[n2], d) for n1, n2, d in G_to_project.edges(data=True)]
    G_base.add_edges_from(edge_to_add)
    return G_base

