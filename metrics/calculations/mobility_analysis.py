import geopandas as gpd
import shapely
import pandas as pd
import json
import shapely.wkt
import networkx as nx

from scipy import spatial
from .errors import ImplementationError
from .base_method import BaseMethod


class AccessibilityIsochrones(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("accessibility_isochrones")
        self.mobility_graph = self.city_model.MobilityGraph.copy()
        self.walk_speed = 4 * 1000 / 60
        self.edge_types = {
            "public_transport": ["subway", "bus", "tram", "trolleybus", "walk"],
            "walk": ["walk"], 
            "drive": ["car"]
            }
        self.travel_names = {
            "public_transport": "Общественный транспорт",
            "walk": "Пешком", 
            "drive": "Личный транспорт"
        }
    
    def get_accessibility_isochrone(self, travel_type, x_from, y_from, weight_value, weight_type, routes=False):
        
        mobility_graph = self.mobility_graph.edge_subgraph(
            [(u, v, k) for u, v, k, d in self.mobility_graph.edges(data=True, keys=True) 
            if d["type"] in self.edge_types[travel_type]]
            )
        nodes_data = pd.DataFrame.from_records(
            [d for u, d in mobility_graph.nodes(data=True)], index=list(mobility_graph.nodes())
            ).sort_index()

        distance, start_node = spatial.KDTree(nodes_data[["x", "y"]]).query([x_from, y_from])
        start_node = nodes_data.iloc[start_node].name
        margin_weight = distance / self.walk_speed if weight_type == "time_min" else distance
        weight_value_remain = weight_value - margin_weight

        weights_sum = nx.single_source_dijkstra_path_length(
            mobility_graph, start_node, cutoff=weight_value_remain, weight=weight_type)
        nodes_data = nodes_data.loc[list(weights_sum.keys())].reset_index()
        nodes_data = gpd.GeoDataFrame(nodes_data, crs=self.city_crs)

        if travel_type == "public_transport" and weight_type == "time_min":
            # 0.8 is routes curvature coefficient 
            distance = dict((k, (weight_value_remain - v) * self.walk_speed * 0.8) for k, v in weights_sum.items())
            nodes_data["left_distance"] = distance.values()
            isochrone_geom = nodes_data["geometry"].buffer(nodes_data["left_distance"])
            isochrone_geom = isochrone_geom.unary_union
        
        else:
            distance = dict((k, (weight_value_remain - v)) for k, v in weights_sum.items())
            nodes_data["left_distance"] = distance.values()
            isochrone_geom = shapely.geometry.MultiPoint(nodes_data["geometry"].tolist()).convex_hull

        isochrone = gpd.GeoDataFrame(
                {"travel_type": [self.travel_names[travel_type]], "weight_type": [weight_type], 
                "weight_value": [weight_value], "geometry": [isochrone_geom]}).set_crs(self.city_crs).to_crs(4326)
 
        routes, stops = self._get_routes(nodes_data, travel_type, weight_type) if routes else (None, None)
        return {"isochrone": json.loads(isochrone.to_json()), "routes": routes, "stops": stops}


    def _get_routes(self, selected_nodes, travel_type, weight_type):

        nodes = selected_nodes[["index", "x", "y", "stop", "desc", "geometry"]]
        subgraph = self.mobility_graph.subgraph(selected_nodes["index"].tolist())
        routes = pd.DataFrame.from_records([
            e[-1] for e in subgraph.edges(data=True, keys=True)
            ]).reset_index(drop=True)

        if travel_type == "public_transport" and weight_type == "time_min":
            stops = nodes[nodes["stop"] == "True"]

            if len(stops) > 0 and len(routes) > 0:
                stops = stops[["index", "x", "y", "geometry", "desc"]]
                stop_types = stops["desc"].apply(
                    lambda x: pd.Series({t: True for t in x.split(", ")}
                    ), type).fillna(False)
                stops = stops.join(stop_types)

                routes_select = routes[routes["type"].isin(self.edge_types[travel_type][:-1])]
                routes_select["geometry"] = routes_select["geometry"].apply(lambda x: shapely.wkt.loads(x))
                routes_select = routes_select[["type", "time_min", "length_meter", "geometry"]]
                routes_select = gpd.GeoDataFrame(routes_select, crs=self.city_crs)
                return json.loads(routes_select.to_crs(4326).to_json()), json.loads(stops.to_crs(4326).to_json())
            else:
                return None, None

        else:
            raise ImplementationError(
                "Route output implemented only with params travel_type='public_transport' and weight_type='time_min'"
                )


class AccessibilityIsochrones_v2(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("accessibility_isochrones_v2")
        self.mobility_graph = self.city_model.MobilityGraph.copy()
        self.graph_nk_time =  self.city_model.graph_nk_time
        self.graph_nk_length = self.city_model.graph_nk_length
        self.walk_speed = 4 * 1000 / 60
        self.edge_types = {
            "public_transport": ["subway", "bus", "tram", "trolleybus", "walk"],
            "walk": ["walk"], 
            "drive": ["car"]
            }
        self.travel_names = {
            "public_transport": "Общественный транспорт",
            "walk": "Пешком", 
            "drive": "Личный транспорт"
        }

    def get_isochrone(self, travel_type, x_from:list, y_from:list, weight_value:int, weight_type, routes=False):

        def _get_nk_distances(nk_dists, loc):
            target_nodes = loc.index.astype('int')
            source_node = loc.name
            distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]
            return pd.Series(data = distances, index = target_nodes)
        
        source = pd.DataFrame(data = list(zip(range(len(x_from)), x_from, y_from)), columns = ['id', 'x', 'y'])
        source = gpd.GeoDataFrame(source, geometry = gpd.points_from_xy(source['x'], source['y'], crs=self.city_crs))
        
        mobility_graph = self.mobility_graph.edge_subgraph(
            [(u, v, k) for u, v, k, d in self.mobility_graph.edges(data=True, keys=True) 
            if d["type"] in self.edge_types[travel_type]
            ])

        mobility_graph = nx.convert_node_labels_to_integers(mobility_graph)
        graph_df = pd.DataFrame.from_dict(dict(mobility_graph.nodes(data=True)), orient='index')
        graph_gdf = gpd.GeoDataFrame(graph_df, geometry = gpd.points_from_xy(graph_df['x'], graph_df['y'])).set_crs(self.city_crs)

        from_services = graph_gdf['geometry'].sindex.nearest(source['geometry'], return_distance = True, return_all = False)

        distances = pd.DataFrame(0, index = from_services[0][1], columns = list(mobility_graph.nodes()))
        
        if weight_type == 'time_min':
            nk_graph = self.graph_nk_time
        elif weight_type == 'length_meter':
            nk_graph = self.graph_nk_length

        nk_dists = nk.distance.SPSP(G = nk_graph, sources = distances.index.values).run()
        distances =  distances.apply(lambda x:_get_nk_distances(nk_dists, x), axis = 1)

        dist_nearest = pd.DataFrame(data = from_services[1], index = from_services[0][1], columns = ['dist'])

        dist_nearest = dist_nearest / self.walk_speed if weight_type == 'time_min' else dist_nearest
        distances = distances.add(dist_nearest.dist, axis = 0)
        
        cols = distances.columns.to_numpy()
        source['isochrone_nodes'] = [cols[x].tolist() for x in distances.le(weight_value).to_numpy()]

        for x, y in list(zip(from_services[0][1], source['isochrone_nodes'])):
            y.extend([x])
        
        source['isochrone_geometry'] = source['isochrone_nodes'].apply(lambda x: [graph_gdf['geometry'].loc[[y for y in x]]])
        source['isochrone_geometry'] = [list(x[0].geometry) for x in source['isochrone_geometry']]
        source['isochrone_geometry'] = [[y.buffer(.01) for y in x] for x in source['isochrone_geometry']]
        source['isochrone_geometry'] = source['isochrone_geometry'].apply(lambda x: shapely.ops.cascaded_union(x).convex_hull)
        source['isochrone_geometry'] = gpd.GeoSeries(source['isochrone_geometry'], crs=self.city_crs)

        isochrones = [gpd.GeoDataFrame(
                {"travel_type": [self.travel_names[travel_type]], "weight_type": [weight_type], 
                "weight_value": [weight_value], "geometry": [x]}, geometry = [x], crs = self.city_crs).to_crs(4326) 
                for x in source['isochrone_geometry']]

        isochrones = pd.concat([x for x in isochrones])
        
        stops, routes = self.get_routes(graph_gdf, source['isochrone_nodes'], travel_type, weight_type) if routes else ([None], [None])
        
        return {"isochrone": json.loads(isochrones.to_json()), "routes": routes, "stops": stops}

    def get_routes(self, graph_gdf, selected_nodes, travel_type, weight_type):
        
        if travel_type == 'public_transport' and weight_type == 'time_min':
            stops = graph_gdf[graph_gdf["stop"] == "True"]
            stop_types = stops["desc"].apply(
                    lambda x: pd.Series({t: True for t in x.split(", ")}
                    ), type).fillna(False)
            stops = stops.join(stop_types)
            stops_result = [stops.loc[stops['nodeID'].isin(x)].to_crs(4326) for x in selected_nodes]
            
            nodes = [x['nodeID'] for x in stops_result]
            subgraph = [self.mobility_graph.subgraph(x) for x in nodes]
            routes = [pd.DataFrame.from_records([e[-1] for e in x.edges(data=True, keys=True)]) for x in subgraph]

            def routes_selection (routes):
                if len(routes) > 0:
                    routes_select = routes[routes["type"].isin(self.edge_types['public_transport'][:-1])]
                    routes_select["geometry"] = routes_select["geometry"].apply(lambda x: shapely.wkt.loads(x))
                    routes_select = routes_select[["type", "time_min", "length_meter", "geometry"]]
                    routes_select = gpd.GeoDataFrame(routes_select, crs=32636).to_crs(4326)
                    return json.loads(routes_select.to_json())
                else:
                    return None

            routes_result = list(map(routes_selection, routes))

            return [json.loads(x.to_json()) for x in stops_result], routes_result

        else:
            raise ImplementationError(
                "Route output implemented only with params travel_type='public_transport' and weight_type='time_min'"
                )