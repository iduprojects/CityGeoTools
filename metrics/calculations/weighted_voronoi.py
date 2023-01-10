import geopandas as gpd
import shapely
import pandas as pd
import math
import json
import shapely.wkt

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
 
        routes, stops = self.get_routes(nodes_data, travel_type, weight_type) if routes else (None, None)
        return {"isochrone": json.loads(isochrone.to_json()), "routes": routes, "stops": stops}


    def get_routes(self, selected_nodes, travel_type, weight_type):

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


class WeightedVoronoi(BaseMethod):

    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)

    @staticmethod
    def _self_weight_list_calculation(start_value, iter_count): 
        log_r = [start_value]
        self_weigth =[]
        max_value = log_r[0] * iter_count
        for i in range(iter_count):
            next_value = log_r[-1] + math.log(max_value / log_r[-1], 1.5)
            log_r.append(next_value)
            self_weigth.append(log_r[-1] - log_r[i])
        return self_weigth, log_r

    @staticmethod
    def _vertex_checker(x_coords, y_coords, growth_rules, encounter_indexes, input_geojson):
        for i in range(len(growth_rules)):
            if growth_rules[i] == False:
                pass
            else:
                for index in encounter_indexes:
                    if shapely.geometry.Point(x_coords[i],y_coords[i]).within(input_geojson['geometry'][index]):
                        growth_rules[i] = False
                        break
        return growth_rules

    @staticmethod
    def _growth_funtion_x(x_coords, growth_rules, iteration_weight):
        growth_x = [x_coords[i-1] + iteration_weight  *math.sin(2 * math.pi * i / 65) 
        if growth_rules[i-1] == True else x_coords[i-1] for i in range(1, len(x_coords) + 1)]
        return growth_x 
    
    @staticmethod
    def _growth_funtion_y(y_coords, growth_rules, iteration_weight):    
        growth_y = [y_coords[i-1] + iteration_weight * math.cos(2 * math.pi * i / 65) 
        if growth_rules[i-1] == True else y_coords[i-1] for i in range(1, len(y_coords) + 1)]
        return growth_y

    def get_weighted_voronoi_result(self, geojson):

        iter_count = 300
        geojson_crs = geojson["crs"]["properties"]["name"]
        input_geojson = gpd.GeoDataFrame.from_features(geojson['features']).set_crs(geojson_crs)
        input_geojson['init_centroid'] = input_geojson.apply(lambda x: list(x['geometry'].coords)[0], axis = 1)
        input_geojson['geometry'] = input_geojson.apply(lambda x: shapely.geometry.Polygon([
            [list(x['geometry'].coords)[0][0] + x['weight'] * math.sin(2 * math.pi * i / 65),
            list(x['geometry'].coords)[0][1] + x['weight'] * math.cos(2 * math.pi * i / 65)] 
            for i in range(1, 65)]), axis =1)
        input_geojson['x'] = input_geojson.apply(
            lambda x: list(list(zip(*list(x['geometry'].exterior.coords)))[0]), axis = 1)
        input_geojson['y'] = input_geojson.apply(
            lambda x: list(list(zip(*list(x['geometry'].exterior.coords)))[1]), axis = 1)
        input_geojson['self_weight'] = input_geojson.apply(
            lambda x: self._self_weight_list_calculation(x['weight'], iter_count)[0], axis = 1)
        input_geojson['self_radius'] = input_geojson.apply(
            lambda x: self._self_weight_list_calculation(x['weight'], iter_count)[1], axis = 1)
        input_geojson['vertex_growth_allow_rule'] = input_geojson.apply(
            lambda x: [True for x in range(len(x['x']))], axis = 1)
        temp = pd.DataFrame({'x':input_geojson.apply(
            lambda x: self._growth_funtion_x(x['x'], x['vertex_growth_allow_rule'],x['self_radius'][-1]), axis = 1),
                    'y':input_geojson.apply(
                        lambda x: self._growth_funtion_y(x['y'], x['vertex_growth_allow_rule'], x['self_radius'][-1]), 
                        axis = 1)}).apply(
                            lambda x: shapely.geometry.Polygon(tuple(zip(x['x'], x['y']))), axis = 1)
        input_geojson['encounter_rule_index'] = [
            [y for y in range(len(temp)) if y != x if temp[x].intersects(temp[y])] for x in range(len(temp))]
        for i in range(iter_count):
            input_geojson['x'] = input_geojson.apply(
                lambda x: self._growth_funtion_x(x['x'], x['vertex_growth_allow_rule'],x['self_weight'][i]), axis = 1)
            input_geojson['y'] = input_geojson.apply(
                lambda x: self._growth_funtion_y(x['y'],x['vertex_growth_allow_rule'],x['self_weight'][i]), axis = 1)
            input_geojson['geometry'] = input_geojson.apply(
                lambda x: shapely.geometry.Polygon(tuple(zip(x['x'], x['y']))), axis = 1)   
            input_geojson['vertex_growth_allow_rule'] = input_geojson.apply(
                lambda x: self._vertex_checker(
                    x['x'], x['y'], x['vertex_growth_allow_rule'], x['encounter_rule_index'], input_geojson), 
                    axis = 1)
        
        start_points = gpd.GeoDataFrame.from_features(geojson['features'])
        x = [list(p.coords)[0][0] for p in start_points['geometry']]
        y = [list(p.coords)[0][1] for p in start_points['geometry']]
        centroid = shapely.geometry.Point(
            (sum(x) / len(start_points['geometry']), sum(y) / len(start_points['geometry'])))
        buffer_untouch = centroid.buffer(start_points.distance(shapely.geometry.Point(centroid)).max()*1.4)
        buffer_untouch = gpd.GeoDataFrame(data = {'id':[1]} ,geometry = [buffer_untouch]).set_crs(3857)
        
        result = gpd.overlay(buffer_untouch, input_geojson, how='difference')
        input_geojson = input_geojson.to_crs(4326)
        result = result.to_crs(4326)
        return {'voronoi_polygons': json.loads(input_geojson[['weight','geometry']].to_json()),
                'deficit_zones': json.loads(result.to_json())}