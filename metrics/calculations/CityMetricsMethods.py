from multiprocessing.sharedctypes import Value
import geopandas as gpd
import shapely
import pandas as pd
import math
import json
import numpy as np
import shapely.wkt
import io
import pca
import networkx as nx
import networkit as nk

from jsonschema.exceptions import ValidationError
from .utils import nk_routes_between_two_points
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy import spatial

class BaseMethod():

    def __init__(self, city_model):

        self.city_model = city_model
        self.city_crs = city_model.city_crs
        self.mode = city_model.mode

    def validation(self, method):
        if self.mode == "user_mode":
            if not self.city_model.methods.if_method_available(method):
                bad_layers = self.city_model.methods.get_bad_layers(method)
                raise ValidationError(f'Layers {", ".join(bad_layers)} do not match specification.')

    @staticmethod
    def get_territorial_select(area_type, area_id, *args):
        return tuple(df[df[area_type + "_id"] == area_id] for df in args)

    @staticmethod
    def get_custom_polygon_select(geojson, set_crs, *args):

        geojson_crs = geojson["crs"]["properties"]["name"]
        geojson = gpd.GeoDataFrame.from_features(geojson['features'])
        geojson = geojson.set_crs(geojson_crs).to_crs(set_crs)
        custom_polygon = geojson['geometry'][0]
        return tuple(df[df.within(custom_polygon)] for df in args)

    # TODO: add method for slicing object's dataframe with specifed parameter

# ########################################  Trafiics calculation  ####################################################
class TrafficCalculator(BaseMethod):

    def __init__(self, city_model):

        BaseMethod.__init__(self, city_model)
        super().validation("traffic_calculator")
        self.stops = self.city_model.PublicTransportStops.copy()
        self.buildings = self.city_model.Buildings.copy()
        self.mobility_graph = self.city_model.graph_nk_length
        self.mobility_graph_attrs = self.city_model.nk_attrs.copy()

    def get_trafic_calculation(self, request_area_geojson):

        living_buildings = self.buildings[self.buildings['population'] > 0]
        living_buildings = living_buildings[['id', 'population', 'geometry']]
        selected_buildings = self.get_custom_polygon_select(request_area_geojson, self.city_crs, living_buildings)[0]

        if len(selected_buildings) == 0:
            return None
        
        stops = self.stops.set_index("id")
        selected_buildings['nearest_stop_id'] = selected_buildings.apply(
            lambda x: stops['geometry'].distance(x['geometry']).idxmin(), axis=1)
        nearest_stops = stops.loc[list(selected_buildings['nearest_stop_id'])]
        path_info = selected_buildings.apply(
            lambda x: nk_routes_between_two_points(self.mobility_graph, self.mobility_graph_attrs,
            p1 = x['geometry'].centroid.coords[0], p2 = stops.loc[x['nearest_stop_id']].geometry.coords[0]), 
            result_type="expand", axis=1)
        house_stop_routes = selected_buildings.copy().drop(["geometry"], axis=1).join(path_info)

        # 30% aprox value of Public transport users
        house_stop_routes['population'] = (house_stop_routes['population'] * 0.3).round().astype("int")
        house_stop_routes = house_stop_routes.rename(
            columns={'population': 'route_traffic', 'id': 'building_id', "route_geometry": "geometry"})
        house_stop_routes = gpd.GeoDataFrame(house_stop_routes, crs=selected_buildings.crs)

        return {"buildings": json.loads(selected_buildings.reset_index(drop=True).to_crs(4326).to_json()), 
                "stops": json.loads(nearest_stops.reset_index(drop=True).to_crs(4326).to_json()), 
                "routes": json.loads(house_stop_routes.reset_index(drop=True).to_crs(4326).to_json())}

# ########################################  Visibility analysis  ####################################################

class VisibilityAnalysis(BaseMethod):

    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("traffic_calculator")
        self.buildings = self.city_model.Buildings.copy()

    def get_visibility_result(self, point, view_distance):
        
        point_buffer = shapely.geometry.Point(point).buffer(view_distance)
        s = self.buildings.within(point_buffer)
        buildings_in_buffer = self.buildings.loc[s[s].index].reset_index(drop=True)
        buffer_exterior_ = list(point_buffer.exterior.coords)
        line_geometry = [shapely.geometry.LineString([point, ext]) for ext in buffer_exterior_]
        buffer_lines_gdf = gpd.GeoDataFrame(geometry=line_geometry)
        united_buildings = buildings_in_buffer.unary_union

        if united_buildings:
            splited_lines = buffer_lines_gdf.apply(lambda x: x['geometry'].difference(united_buildings), axis=1)
        else:
            splited_lines = buffer_lines_gdf["geometry"]

        splited_lines_gdf = gpd.GeoDataFrame(geometry=splited_lines).explode()
        splited_lines_list = []

        for u, v in splited_lines_gdf.groupby(level=0):
            splited_lines_list.append(v.iloc[0]['geometry'].coords[-1])
        circuit = shapely.geometry.Polygon(splited_lines_list)
        if united_buildings:
            circuit = circuit.difference(united_buildings)

        view_zone = gpd.GeoDataFrame(geometry=[circuit]).set_crs(self.city_crs).to_crs(4326)
        return json.loads(view_zone.to_json())

# ########################################  Weighted Voronoi  ####################################################
class WeightedVoronoi(BaseMethod):

    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)

    @staticmethod
    def self_weight_list_calculation(start_value, iter_count): 
        log_r = [start_value]
        self_weigth =[]
        max_value = log_r[0] * iter_count
        for i in range(iter_count):
            next_value = log_r[-1] + math.log(max_value / log_r[-1], 1.5)
            log_r.append(next_value)
            self_weigth.append(log_r[-1] - log_r[i])
        return self_weigth, log_r

    @staticmethod
    def vertex_checker(x_coords, y_coords, growth_rules, encounter_indexes, input_geojson):
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
    def growth_funtion_x(x_coords, growth_rules, iteration_weight):
        growth_x = [x_coords[i-1] + iteration_weight  *math.sin(2 * math.pi * i / 65) 
        if growth_rules[i-1] == True else x_coords[i-1] for i in range(1, len(x_coords) + 1)]
        return growth_x 
    
    @staticmethod
    def growth_funtion_y(y_coords, growth_rules, iteration_weight):    
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
            lambda x: self.self_weight_list_calculation(x['weight'], iter_count)[0], axis = 1)
        input_geojson['self_radius'] = input_geojson.apply(
            lambda x: self.self_weight_list_calculation(x['weight'], iter_count)[1], axis = 1)
        input_geojson['vertex_growth_allow_rule'] = input_geojson.apply(
            lambda x: [True for x in range(len(x['x']))], axis = 1)
        temp = pd.DataFrame({'x':input_geojson.apply(
            lambda x: self.growth_funtion_x(x['x'], x['vertex_growth_allow_rule'],x['self_radius'][-1]), axis = 1),
                    'y':input_geojson.apply(
                        lambda x: self.growth_funtion_y(x['y'], x['vertex_growth_allow_rule'], x['self_radius'][-1]), 
                        axis = 1)}).apply(
                            lambda x: shapely.geometry.Polygon(tuple(zip(x['x'], x['y']))), axis = 1)
        input_geojson['encounter_rule_index'] = [
            [y for y in range(len(temp)) if y != x if temp[x].intersects(temp[y])] for x in range(len(temp))]
        for i in range(iter_count):
            input_geojson['x'] = input_geojson.apply(
                lambda x: self.growth_funtion_x(x['x'], x['vertex_growth_allow_rule'],x['self_weight'][i]), axis = 1)
            input_geojson['y'] = input_geojson.apply(
                lambda x: self.growth_funtion_y(x['y'],x['vertex_growth_allow_rule'],x['self_weight'][i]), axis = 1)
            input_geojson['geometry'] = input_geojson.apply(
                lambda x: shapely.geometry.Polygon(tuple(zip(x['x'], x['y']))), axis = 1)   
            input_geojson['vertex_growth_allow_rule'] = input_geojson.apply(
                lambda x: self.vertex_checker(
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

# ########################################  Blocks clusterization  ###################################################
class BlocksClusterization(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("blocks_clusterization")
        self.services = self.city_model.Services.copy()
        self.blocks = self.city_model.Blocks.copy()
    
    def clusterize(self, service_types):

        service_in_blocks = self.services.groupby(["block_id", "service_code"])["id"].count().unstack(fill_value=0)
        without_services = self.blocks["id"][~self.blocks["id"].isin(service_in_blocks.index)].values
        without_services = pd.DataFrame(columns=service_in_blocks.columns, index=without_services).fillna(0)
        service_in_blocks = pd.concat([without_services, service_in_blocks])

        service_in_blocks = service_in_blocks[service_types]
        clusterization = linkage(service_in_blocks, method="ward")

        return clusterization, service_in_blocks

    @staticmethod
    def get_clusters_number(clusterization):

        distance = clusterization[-100:, 2]
        clusters = np.arange(1, len(distance) + 1)
        acceleration = np.diff(distance, 2)[::-1]
        series_acceleration = pd.Series(acceleration, index=clusters[:-2] + 1)

        # There are always more than two clusters
        series_acceleration = series_acceleration.iloc[1:]
        clusters_number = series_acceleration.idxmax()

        return clusters_number

    def get_blocks(self, service_types, clusters_number=None, area_type=None, area_id=None, geojson=None):

        clusterization, service_in_blocks = self.clusterize(service_types)
        
        # If user doesn't specified the number of clusters, use default value.
        # The default value is determined with the rate of change in the distance between clusters
        if not clusters_number:
            clusters_number = self.get_clusters_number(clusterization)

        service_in_blocks["cluster_labels"] = fcluster(clusterization, t=int(clusters_number), criterion="maxclust")
        blocks = self.blocks.join(service_in_blocks, on="id")
        mean_services_number = service_in_blocks.groupby("cluster_labels")[service_types].mean().round()
        mean_services_number = service_in_blocks[["cluster_labels"]].join(mean_services_number, on="cluster_labels")
        deviations_services_number = service_in_blocks[service_types] - mean_services_number[service_types]
        blocks = blocks.join(deviations_services_number, on="id", rsuffix="_deviation")

        if area_type and area_id:
            blocks = self.get_territorial_select(area_type, area_id, blocks)[0]
        elif geojson:
            blocks = self.get_custom_polygon_select(geojson, self.city_crs, blocks)[0]

        return json.loads(blocks.to_crs(4326).to_json())

    def get_dendrogram(self, service_types):
            
            clusterization, service_in_blocks = self.clusterize(service_types)

            img = io.BytesIO()
            plt.figure(figsize=(20, 10))
            plt.title("Dendrogram")
            plt.xlabel("Distance")
            plt.ylabel("Block clusters")
            dn = dendrogram(clusterization, p=7, truncate_mode="level")
            plt.savefig(img, format="png")
            plt.close()
            img.seek(0)

            return img

# ########################################  Services clusterization  #################################################
class ServicesClusterization(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("services_clusterization")
        self.services = self.city_model.Services.copy()
    
    @staticmethod
    def get_service_cluster(services_select, condition, condition_value):
        services_coords = pd.DataFrame({"x": services_select.geometry.x, "y": services_select.geometry.y})
        clusterization = linkage(services_coords.to_numpy(), method="ward")
        services_select["cluster"] = fcluster(clusterization, t=condition_value, criterion=condition)
        return services_select

    @staticmethod
    def find_dense_groups(loc, n_std):
        if len(loc) > 1:
            X = pd.DataFrame({"x": loc.x, "y": loc.y})
            X = X.to_numpy()
            outlier = pca.spe_dmodx(X, n_std=n_std)[0]["y_bool_spe"]
            return pd.Series(data=outlier.values, index=loc.index)
        else:
            return pd.Series(data=True, index=loc.index)

    @staticmethod
    def get_service_ratio(loc):
        all_services = loc["id"].count()
        services_count = loc.groupby("service_code")["id"].count()
        return (services_count / all_services).round(2)

    def get_clusters_polygon(self, service_types, area_type = None, area_id = None, geojson = None, 
                            condition="distance", condition_value=4000, n_std = 2):

        services_select = self.services[self.services["service_code"].isin(service_types)]
        if area_type and area_id:
            services_select = self.get_territorial_select(area_type, area_id, services_select)[0]
        elif geojson:
            services_select = self.get_custom_polygon_select(geojson, self.city_crs, services_select)[0]
        if len(services_select) <= 1:
            return None

        services_select = self.get_service_cluster(services_select, condition, condition_value)

        # Find outliers of clusters and exclude it
        outlier = services_select.groupby("cluster")["geometry"].apply(lambda x: self.find_dense_groups(x, n_std))
        if any(~outlier):
            services_normal = services_select[~outlier]

            if len(services_normal) > 0:
                cluster_service = services_normal.groupby(["cluster"]).apply(lambda x: self.get_service_ratio(x))
                if isinstance(cluster_service, pd.Series):
                    cluster_service = cluster_service.unstack(level=1, fill_value=0)

                # Get MultiPoint from cluster Points and make polygon
                polygons_normal = services_normal.dissolve("cluster").convex_hull
                df_clusters_normal = pd.concat([cluster_service, polygons_normal.rename("geometry")], axis=1
                                                )
                cluster_normal = df_clusters_normal.index.max()
        else:
            df_clusters_normal = None

        # Select outliers 
        if any(outlier):
            services_outlier = services_select[outlier]

            # Reindex clusters
            clusters_outlier = cluster_normal + 1
            new_clusters = [c for c in range(clusters_outlier, clusters_outlier + len(services_outlier))]
            services_outlier["cluster"] = new_clusters
            cluster_service = services_outlier.groupby(["cluster"]).apply(lambda x: self.get_service_ratio(x))
            if isinstance(cluster_service, pd.Series):
                cluster_service = cluster_service.unstack(level=1, fill_value=0)
            df_clusters_outlier = cluster_service.join(services_outlier.set_index("cluster")["geometry"])
        else:
            df_clusters_outlier = None

        df_clusters = pd.concat([df_clusters_normal, df_clusters_outlier]).fillna(0).set_geometry("geometry")
        df_clusters["geometry"] = df_clusters["geometry"].buffer(50, join_style=3)
        df_clusters = df_clusters.rename(columns={"index": "cluster_id"})

        services = pd.concat([services_normal, services_outlier]).set_crs(self.city_crs).to_crs(4326)
        df_clusters = df_clusters.set_crs(self.city_crs).to_crs(4326)

        return {"polygons": json.loads(df_clusters.to_json()), "services": json.loads(services.to_json())}

# #############################################  Spacematrix  #######################################################
class Spacematrix(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("spacematrix")
        self.buildings = self.city_model.Buildings.copy()
        self.blocks = self.city_model.Blocks.copy().set_index("id")

    @staticmethod
    def simple_preprocess_data(buildings, blocks):

        # temporary filters. since there are a few bugs in buildings table from DB
        buildings = buildings[buildings["block_id"].notna()]
        buildings = buildings[buildings["storeys_count"].notna()]
        buildings["is_living"] = buildings["is_living"].fillna(False)

        buildings["building_area"] = buildings["basement_area"] * buildings["storeys_count"]
        bad_living_area = buildings[buildings["living_area"] > buildings["building_area"]].index
        buildings.loc[bad_living_area, "living_area"] = None

        living_grouper = buildings.groupby(["is_living"])
        buildings["living_area"] = living_grouper.apply(
            lambda x: x.living_area.fillna(x.building_area * 0.8) if x.name else x.living_area.fillna(0)
            ).droplevel(0).round(2)

        blocks_area_nans = blocks[blocks["area"].isna()].index
        blocks.loc[blocks_area_nans, "area"] = blocks["geometry"].loc[blocks_area_nans].area

        return buildings, blocks

    @staticmethod
    def calculate_block_indices(buildings, blocks):

        sum_grouper = buildings.groupby(["block_id"]).sum()
        blocks["FSI"] = sum_grouper["building_area"] / blocks["area"]
        blocks["GSI"] = sum_grouper["basement_area"] / blocks["area"]
        blocks["MXI"] = (sum_grouper["living_area"] / sum_grouper["building_area"]).round(2)
        blocks["L"] =( blocks["FSI"] / blocks["GSI"]).round()
        blocks["OSR"] = ((1 - blocks["GSI"]) / blocks["FSI"]).round(2)
        blocks[["FSI", "GSI"]] = blocks[["FSI", "GSI"]].round(2)

        return blocks

    @staticmethod
    def name_spacematrix_morph_types(cluster):

        ranges = [[0, 3, 6, 10, 17], 
                  [0, 1, 2], 
                  [0, 0.22, 0.55]]

        labels = [["Малоэтажный", "Среднеэтажный", "Повышенной этажности", "Многоэтажный", "Высотный"],
                  [" низкоплотный", "", " плотный"], 
                  [" нежилой", " смешанный", " жилой"]]

        cluster_name = []
        for ind in range(len(cluster)):
            cluster_name.append(
                labels[ind][[i for i in range(len(ranges[ind])) if cluster.iloc[ind] >= ranges[ind][i]][-1]]
                )
        return "".join(cluster_name)


    def get_spacematrix_morph_types(self, clusters_number=11, area_type=None, area_id=None, geojson=None):

        buildings, blocks = self.simple_preprocess_data(self.buildings, self.blocks)
        blocks = self.calculate_block_indices(buildings, blocks)

        # blocks with OSR >=10 considered as unbuilt blocks
        X = blocks[blocks["OSR"] < 10][['FSI', 'L', 'MXI']].dropna()
        scaler = StandardScaler()
        X_scaler = pd.DataFrame(scaler.fit_transform(X))
        kmeans = KMeans(n_clusters=clusters_number, random_state=42).fit(X_scaler)
        X["spacematrix_cluster"] = kmeans.labels_
        blocks = blocks.join(X["spacematrix_cluster"])
        cluster_grouper = blocks.groupby(["spacematrix_cluster"]).median()
        named_clusters = cluster_grouper[["L", "FSI", "MXI"]].apply(
            lambda x: self.name_spacematrix_morph_types(x), axis=1)
        blocks = blocks.join(named_clusters.rename("spacematrix_morphotype"), on="spacematrix_cluster")

        if area_type and area_id:
            blocks = self.get_territorial_select(area_type, area_id, blocks)[0]
        elif geojson:
            blocks = self.get_custom_polygon_select(geojson, self.city_crs, blocks)[0]

        return json.loads(blocks.to_crs(4326).to_json())

# ######################################### Accessibility isochrones #################################################
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
 
        routes, stops = self.get_routes(nodes_data, travel_type) if routes else (None, None)
        return {"isochrone": json.loads(isochrone.to_json()), "routes": routes, "stops": stops}


    def get_routes(self, selected_nodes, travel_type):

        nodes = selected_nodes[["index", "x", "y", "stop", "desc", "geometry"]]
        subgraph = self.mobility_graph.subgraph(selected_nodes["index"].tolist())
        routes = pd.DataFrame.from_records([
            e[-1] for e in subgraph.edges(data=True, keys=True)
            ]).reset_index(drop=True)

        if travel_type == "public_transport":
            stops = nodes[nodes["stop"] == "True"]
            stops = stops[["index", "x", "y", "geometry", "desc"]]
            stop_types = stops["desc"].apply(
                lambda x: pd.Series({t: True for t in x.split(", ")}
                ), type).fillna(False)
            stops = stops.join(stop_types)

            routes = routes[routes["type"].isin(self.edge_types[travel_type][:-1])]
        
        else:
            raise ValidationError("Not implementet yet.")
        
        routes["geometry"] = routes["geometry"].apply(lambda x: shapely.wkt.loads(x))
        routes = routes[["type", "time_min", "length_meter", "geometry"]]
        routes = gpd.GeoDataFrame(routes, crs=self.city_crs)

        return json.loads(routes.to_crs(4326).to_json()), json.loads(stops.to_crs(4326).to_json())


# ################################################ Diversity ######################################################
class Diversity(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("diversity")
        self.mobility_graph_length = self.city_model.graph_nk_length
        self.mobility_graph_time = self.city_model.graph_nk_time
        self.graph_attrs = self.city_model.nk_attrs.copy()
        self.buildings = self.city_model.Buildings.copy()
        self.services = self.city_model.Services.copy()
        self.service_types = self.city_model.ServiceTypes.copy()
        self.municipalities = self.city_model.Municipalities.copy()
        self.blocks = self.city_model.Blocks.copy()

    def define_service_normative(self, service_type):

        service_type_info = self.service_types[self.service_types["code"] == service_type]

        if service_type_info["walking_radius_normative"].notna().values:
            travel_type = "walk"
            weigth = "length_meter"
            limit_value = service_type_info["walking_radius_normative"].values[0]
            graph = self.mobility_graph_length
        elif service_type_info["public_transport_time_normative"].notna().values:
            travel_type = "public_transport"
            weigth = "time_min"
            limit_value = service_type_info["public_transport_time_normative"].values[0]
            graph = self.mobility_graph_time
        else:
            raise ValueError("Any service type normative is None.")

        return travel_type, weigth, limit_value, graph
        
    def get_distance_matrix(self, houses, services, graph, limit_value):

        nodes_data = pd.DataFrame(self.graph_attrs.values(), index=self.graph_attrs.keys())
        houses_distance, houses_nodes = spatial.cKDTree(nodes_data).query([houses[["x", "y"]]])
        services_distance, services_nodes = spatial.cKDTree(nodes_data).query([services[["x", "y"]]])

        if len(services_nodes[0]) < len(houses_nodes[0]):
            source, target = services_nodes, houses_nodes
            source_dist, target_dist = services_distance, houses_distance
        else:
            source, target = houses_nodes, services_nodes
            source_dist, target_dist = houses_distance, services_distance

        dijkstra = nk.distance.SPSP(graph, source[0])
        dijkstra = dijkstra.run()
        dist_matrix = dijkstra.getDistances(asarray=True)
        dist_matrix = dist_matrix[:, target[0]] + target_dist[0] + np.vstack(np.array(source_dist[0]))
        dist_matrix = np.where(dist_matrix > limit_value, dist_matrix, 1)
        dist_matrix = np.where(dist_matrix <= limit_value, dist_matrix, 0)

        return dist_matrix

    @staticmethod
    def calculate_diversity(houses, dist_matrix):

        count_services = dist_matrix.sum(axis=0)
        diversity_estim = {1:0.2, 2:0.4, 3:0.6, 4:0.8, 5:1}
        for count, estim in diversity_estim.items():
            count_services[count_services == count] = estim
        count_services = np.where(count_services < 5, count_services, 1)
        houses_diversity = pd.Series(count_services, index=houses["id"]).rename("diversity")
        houses = houses.join(houses_diversity, on="id")
        return houses

    def get_diversity(self, service_type):

        services = self.services[self.services["service_code"] == service_type]
        houses = self.buildings[self.buildings['is_living'] == True].reset_index(drop=True)

        travel_type, weigth, limit_value, graph = self.define_service_normative(service_type)
        dist_matrix = self.get_distance_matrix(houses, services, graph, limit_value)
        houses = self.calculate_diversity(houses, dist_matrix)

        blocks = self.blocks.dropna(subset=["municipality_id"]) # TEMPORARY
        blocks = self.blocks.join(houses.groupby(["block_id"])["diversity"].mean().round(2), on="id")
        municipalities = self.municipalities.join(
            houses.groupby(["municipality_id"])["diversity"].mean().round(2), on="id"
            )
        return {
            "municipalities": json.loads(municipalities.to_crs(4326).fillna("None").to_json()),
            "blocks": json.loads(blocks.to_crs(4326).fillna("None").to_json())
                    }

    def get_houses(self, block_id, service_type):

        services = self.services[self.services["service_code"] == service_type]
        houses = self.buildings[self.buildings['is_living'] == True]
        houses_in_block = self.buildings[self.buildings['block_id'] == block_id].reset_index(drop=True)

        travel_type, weigth, limit_value, graph = self.define_service_normative(service_type)
        dist_matrix = self.get_distance_matrix(houses_in_block, services, graph, limit_value)
        houses = self.calculate_diversity(houses_in_block,np.transpose(dist_matrix))

        return json.loads(houses_in_block.to_crs(4326).to_json())

    def get_info(self, house_id, service_type):

        services = self.services[self.services["service_code"] == service_type]
        house = self.buildings[self.buildings['id'] == house_id].reset_index(drop=True)
        house_x, house_y = house[["x", "y"]].values[0]

        travel_type, weigth, limit_value, graph = self.define_service_normative(service_type)
        dist_matrix = self.get_distance_matrix(house, services, graph, limit_value)
        house = self.calculate_diversity(house, np.vstack(dist_matrix[0]))

        selected_services = services[dist_matrix[0] == 1]
        isochrone = AccessibilityIsochrones(self.city_model).get_accessibility_isochrone(
            travel_type, house_x, house_y, limit_value, weigth
        )
        return {
            "house": json.loads(house.to_crs(4326).to_json()),
            "services": json.loads(selected_services.to_crs(4326).to_json()),
            "isochrone": isochrone["isochrone"]
        }



                        
