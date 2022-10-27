from typing import Any, Optional
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
import pandas.core.indexes as pd_index
import pulp
from sqlalchemy import create_engine

from jsonschema.exceptions import ValidationError
from .utils import nk_routes_between_two_points
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy import spatial
from .errors import TerritorialSelectError, SelectedValueError, ImplementationError
from itertools import product

#from app.schemas import FeatureCollectionWithCRS


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
    def get_custom_polygon_select(geojson: dict, set_crs, *args):
        geojson_crs = geojson["crs"]["properties"]["name"]
        geojson = gpd.GeoDataFrame.from_features(geojson['features'])
        geojson = geojson.set_crs(geojson_crs).to_crs(set_crs)
        custom_polygon = geojson['geometry'][0]
        return tuple(df[df.within(custom_polygon)] for df in args)

    # TODO: add method for slicing object's dataframe with specifed parameter

#########################################  Trafiics calculation  ####################################################
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
            raise TerritorialSelectError("living buildings")
        
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

        if sum(self.services["service_code"].isin(service_types)) == 0:
            raise SelectedValueError("services", service_types, "service_code")

        service_in_blocks = self.services.groupby(["block_id", "service_code"])["id"].count().unstack(fill_value=0)
        without_services = self.blocks["id"][~self.blocks["id"].isin(service_in_blocks.index)].values
        without_services = pd.DataFrame(columns=service_in_blocks.columns, index=without_services).fillna(0)
        service_in_blocks = pd.concat([without_services, service_in_blocks])
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
            raise TerritorialSelectError("services")

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
            services_outlier = None
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
        living_area = living_grouper.apply(
            lambda x: x.living_area.fillna(x.building_area * 0.8) if x.name else x.living_area.fillna(0)
            )
        if type(living_area.index) == pd_index.multi.MultiIndex:
             buildings["living_area"] = living_area.droplevel(0).round(2)
        else:
            buildings["living_area"] = living_area.values[0].round(2)

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
            if area_type == "block":
                try: 
                    blocks = blocks.loc[[area_id]]
                except:
                    raise SelectedValueError("build-up block", "area_id", "id")
            else:
                blocks = self.get_territorial_select(area_type, area_id, blocks)[0]
        elif geojson:
            blocks = self.get_custom_polygon_select(geojson, self.city_crs, blocks)[0]

        return json.loads(blocks.reset_index().to_crs(4326).to_json())

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

                routes_select = routes["type"].isin(self.edge_types[travel_type][:-1])
                routes["geometry"] = routes["geometry"].apply(lambda x: shapely.wkt.loads(x))
                routes = routes[["type", "time_min", "length_meter", "geometry"]]
                routes = gpd.GeoDataFrame(routes, crs=self.city_crs)
                return json.loads(routes.to_crs(4326).to_json()), json.loads(stops.to_crs(4326).to_json())
            else:
                return None, None

        else:
            raise ImplementationError(
                "Route output implemented only with params travel_type='public_transport' and weight_type='time_min'"
                )


# ################################################ Diversity ######################################################
class Diversity(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("diversity")
        self.mobility_graph_length = self.city_model.graph_nk_length
        self.mobility_graph_time = self.city_model.graph_nk_time
        self.graph_attrs = self.city_model.nk_attrs.copy()
        self.services = self.city_model.Services.copy()
        self.service_types = self.city_model.ServiceTypes.copy()
        self.municipalities = self.city_model.Municipalities.copy()
        self.blocks = self.city_model.Blocks.copy()

        self.buildings = self.city_model.Buildings.copy()
        self.living_buildings = self.buildings[self.buildings['is_living'] == True].reset_index(drop=True)
        if len(self.living_buildings) == 0:
            raise TerritorialSelectError("living buildings")

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

        houses_distance, houses_nodes = spatial.cKDTree(self.graph_attrs).query([houses[["x", "y"]]])
        services_distance, services_nodes = spatial.cKDTree(self.graph_attrs).query([services[["x", "y"]]])

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
        if len(services) == 0:
            raise SelectedValueError("services", service_type, "service_code")
        houses = self.living_buildings

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
        if len(services) == 0:
            raise SelectedValueError("services", service_type, "service_code")

        houses_in_block = self.living_buildings[self.living_buildings['block_id'] == block_id].reset_index(drop=True)
        if len(houses_in_block) == 0:
            raise TerritorialSelectError("living buildings")

        travel_type, weigth, limit_value, graph = self.define_service_normative(service_type)
        dist_matrix = self.get_distance_matrix(houses_in_block, services, graph, limit_value)
        dist_matrix = np.transpose(dist_matrix) if len(houses_in_block) <= len(services) else dist_matrix
        houses_in_block = self.calculate_diversity(houses_in_block, dist_matrix)

        return json.loads(houses_in_block.to_crs(4326).to_json())

    def get_info(self, house_id, service_type):

        services = self.services[self.services["service_code"] == service_type]
        if len(services) == 0:
            raise SelectedValueError("services", service_type, "service_code")

        house = self.living_buildings[self.living_buildings['id'] == house_id].reset_index(drop=True)
        if len(house) == 0:
            raise SelectedValueError("living building", house_id, "id")
            
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
        

# ######################################### Collocation Matrix #################################################

class CollocationMatrix(BaseMethod):
    def _init_(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("services_clusterization")
        self.services = self.city_model.Services.copy()

    def get_collocation_matrix(self, services):
        services = services.dropna().reset_index(drop=True)[['city_service_type_code','block_id']].sort_values('city_service_type_code')
        services['count'] = 0
        
        collocation_matrix = self.get_numerator(services) / self.get_denominator(services)

        return json.loads(collocation_matrix.to_json())

    def get_numerator(self, services):
        services_numerator = services.pivot_table(index='block_id', columns='city_service_type_code', values='count')

        pairs_services_numerator = [(a, b) for idx, a in enumerate(services_numerator) for b in services_numerator[idx + 1:]]
        pairs_services_numerator = dict.fromkeys(pairs_services_numerator, 0)

        res_numerator = {}
        n_col = len(services_numerator.columns)
        for i in range(n_col):
            for j in range(i + 1, n_col):
                col1 = services_numerator.columns[i]
                col2 = services_numerator.columns[j]
                res_numerator[col1, col2] = sum(services_numerator[col1] == services_numerator[col2])
                res_numerator[col2, col1] = sum(services_numerator[col2] == services_numerator[col1])
        pairs_services_numerator.update(res_numerator)

        numerator = pd.Series(pairs_services_numerator).reset_index(drop=False).set_index(['level_0','level_1']).rename(columns={0:'count'})
        numerator = numerator.pivot_table(index='level_0', columns='level_1', values='count')

        return numerator

    def get_denominator(self, services):
        count_type_block = services.groupby('city_service_type_code')['block_id'].nunique().reset_index(drop=False)

        pairs_services_denominator = []
        for i in product(count_type_block['city_service_type_code'], repeat=2):
            pairs_services_denominator.append(i)

        types_blocks_sum = []
        for i,j in pairs_services_denominator:
            if [i]==[j]:
                    types_blocks_sum.append(0)
            else:
                    num1 = count_type_block.loc[count_type_block['city_service_type_code'] == i, 'block_id'].iloc[0]
                    num2 = count_type_block.loc[count_type_block['city_service_type_code'] == j, 'block_id'].iloc[0]
                    types_blocks_sum.append(num1+num2)

        res_denominator = {}
        for row in range(len(pairs_services_denominator)):
            res_denominator[pairs_services_denominator[row]] = types_blocks_sum[row]

        sum_res_denominator = pd.Series(res_denominator).reset_index(drop=False).set_index(['level_0','level_1']).rename(columns={0:'count'})
        sum_res_denominator = sum_res_denominator.pivot_table(index='level_0', columns='level_1', values='count')

        denominator = sum_res_denominator - self.get_numerator(services)

        return denominator

# ######################################### New Provisions #################################################
           
class City_Provisions(BaseMethod): 

    def __init__(self, city_model: Any, service_types: list, valuation_type: str, year: int,
                 user_provisions: Optional[list], user_changes_buildings: Optional[dict],
                 user_changes_services: Optional[dict], user_selection_zone: Optional[dict]):
        '''
        >>> City_Provisions(city_model,service_types = "kindergartens", valuation_type = "normative", year = 2022).get_provisons()
        >>>
        >>>
        '''
        BaseMethod.__init__(self, city_model)
        self.engine = city_model.engine
        self.service_types = service_types
        self.valuation_type = valuation_type
        self.year = year
        self.city = city_model.city_name
        self.service_types_normatives = city_model.ServiceTypes[city_model.ServiceTypes['code'].isin(service_types)].copy(deep = True)
        self.service_types_normatives.index = self.service_types_normatives['code'].values
        
        self.graph_nk_length = city_model.graph_nk_length
        self.graph_nk_time =  city_model.graph_nk_time
        self.nx_graph =  city_model.MobilityGraph
                                         
        self.buildings = city_model.Buildings.copy(deep = True)
        #self.buildings.index = self.buildings['functional_object_id'].values.astype(int)
        self.buildings.index = range(0, len(self.buildings))

        self.services = city_model.Services[city_model.Services['service_code'].isin(service_types)].copy(deep = True)
        #self.services.index = self.services['id'].values.astype(int)
        self.services.index = range(len(self.buildings) + 1, len(self.buildings) + len(self.services) + 1)


        
        self.user_provisions = {}
        self.user_changes_buildings = {}
        self.user_changes_services = {}
        self.buildings_old_values = None
        self.services_old_values = None

        for service_type in service_types:
            self.demands = pd.read_sql(f'''SELECT functional_object_id, {service_type}_service_demand_value_{self.valuation_type} 
                                        FROM social_stats.buildings_load_future
                                        WHERE year = {self.year}
                                        ''', con = self.engine)
            self.buildings = self.buildings.merge(self.demands, on = 'functional_object_id', how = 'right').dropna()
            self.buildings[f'{service_type}_service_demand_left_value_{self.valuation_type}'] = self.buildings[f'{service_type}_service_demand_value_{self.valuation_type}']
        self.buildings.index = self.buildings['functional_object_id'].values.astype(int)
        
        self.services['capacity_left'] = self.services['capacity']

        self.Provisions = {service_type:{'destination_matrix': None, 
                                         'distance_matrix': None,
                                         'normative_distance':None,
                                         'buildings':None,
                                         'services': None,
                                         'selected_graph':None} for service_type in service_types}
        self.new_Provisions = {service_type:{'destination_matrix': None, 
                                            'distance_matrix': None,
                                            'normative_distance':None,
                                            'buildings':None,
                                            'services': None,
                                            'selected_graph':None} for service_type in service_types}
        #Bad interface , raise error must be 
        if user_changes_services:
            print(2.1)
            self.user_changes_services = gpd.GeoDataFrame.from_features(user_changes_services['features']).set_crs(4326).to_crs(self.city_crs)
            self.user_changes_services.index = self.user_changes_services['id'].values.astype(int)
            self.user_changes_services = self.user_changes_services.combine_first(self.services)
            self.user_changes_services.index = self.user_changes_services['id'].values.astype(int)
            self.user_changes_services['capacity_left'] = self.user_changes_services['capacity']
            self.services_old_values = self.user_changes_services[['capacity','capacity_left','carried_capacity_within','carried_capacity_without']]
            self.user_changes_services = self.user_changes_services.set_crs(self.city_crs)
            #
            self.user_changes_services.index = range(0, len(self.user_changes_services))
        else:
            print(2.2)
            self.user_changes_services = self.services.copy(deep = True)
        
        if user_changes_buildings:
            old_cols = []
            print(1.1)
            self.user_changes_buildings = gpd.GeoDataFrame.from_features(user_changes_buildings['features']).set_crs(4326).to_crs(self.city_crs)
            self.user_changes_buildings.index = self.user_changes_buildings['functional_object_id'].values.astype(int)
            self.user_changes_buildings = self.user_changes_buildings.combine_first(self.buildings)
            self.user_changes_buildings.index = self.user_changes_buildings['functional_object_id'].values.astype(int)
            for service_type in service_types:
                old_cols.extend([f'{service_type}_provison_value', 
                                 f'{service_type}_service_demand_left_value_{self.valuation_type}', 
                                 f'{service_type}_service_demand_value_{self.valuation_type}', 
                                 f'{service_type}_supplyed_demands_within', 
                                 f'{service_type}_supplyed_demands_without'])
                self.user_changes_buildings[f'{service_type}_service_demand_left_value_{self.valuation_type}'] = self.user_changes_buildings[f'{service_type}_service_demand_value_{self.valuation_type}'].values
            self.buildings_old_values = self.user_changes_buildings[old_cols]
            self.user_changes_buildings = self.user_changes_buildings.set_crs(self.city_crs)
            self.user_changes_buildings.index = range(len(self.user_changes_services) + 1, len(self.user_changes_services) + len(self.user_changes_buildings) + 1)

        else:
            print(1.2)
            self.user_changes_buildings = self.buildings.copy()
        if user_provisions:
            for service_type in service_types:
                print(3.1)
                self.user_provisions[service_type]  = pd.DataFrame(0, index =  self.user_changes_services.index.values,
                                                                      columns =  self.user_changes_buildings.index.values)
                self.user_provisions[service_type] = (self.user_provisions[service_type] + self._restore_user_provisions(user_provisions[service_type])).fillna(0)
        else:
            print(3.2)
            self.user_provisions = None

        if user_selection_zone:
            print(4.1)
            gdf = gpd.GeoDataFrame(data = {"id":[1]}, 
                                    geometry = [shapely.geometry.shape(user_selection_zone)],
                                    crs = 4326).to_crs(city_model.city_crs)
            self.user_selection_zone = gdf['geometry'][0]
        else:
            print(4.2)
            self.user_selection_zone = None

    def get_provisions(self, ):

        try:
            self.Provisions = pd.read_sql(f'''SELECT * 
                                         FROM provisions.{self.city}_{self.service_type}_{self.valuation_type}_{self.year}_provisions
                                         ''', con = self.engine)
            self.Matrix = pd.read_sql(f'''SELECT * 
                                     FROM provisions.{self.city}_{self.service_type}_{self.valuation_type}_{self.year}_matrix
                                     ''', con = self.engine)
                #read tables from base and add em to existing buildings and services tables, return.
        except: 
            for service_type in self.service_types:
                
                normative_distance = self.service_types_normatives.loc[service_type].dropna().copy(deep = True)
                try:
                    self.Provisions[service_type]['normative_distance'] = normative_distance['walking_radius_normative']
                    self.Provisions[service_type]['selected_graph'] = self.graph_nk_length
                except:
                    self.Provisions[service_type]['normative_distance'] = normative_distance['public_transport_time_normative']
                    self.Provisions[service_type]['selected_graph'] = self.graph_nk_time
                
                self.Provisions[service_type]['buildings'] = self.buildings.copy(deep = True)
                self.Provisions[service_type]['services'] = self.services[self.services['service_code'] == service_type].copy(deep = True)
                
                self.Provisions[service_type] =  self._calculate_provisions(self.Provisions[service_type], service_type)
                self.Provisions[service_type]['buildings'], self.Provisions[service_type]['services'] = self._additional_options(self.Provisions[service_type]['buildings'].copy(), 
                                                                                                                                 self.Provisions[service_type]['services'].copy(),
                                                                                                                                 self.Provisions[service_type]['distance_matrix'].copy(),
                                                                                                                                 self.Provisions[service_type]['destination_matrix'].copy(),
                                                                                                                                 self.Provisions[service_type]['normative_distance'],
                                                                                                                                 service_type,
                                                                                                                                 self.user_selection_zone,
                                                                                                                                 self.valuation_type)
            cols_to_drop = [x for x in self.buildings.columns for service_type in self.service_types if service_type in x]
            self.buildings = self.buildings.drop(columns = cols_to_drop)
            for service_type in self.service_types:
                self.buildings = self.buildings.merge(self.Provisions[service_type]['buildings'], 
                                                      left_on = 'functional_object_id', 
                                                      right_on = 'functional_object_id')
                                                                          #left_index=True, 
                                                                          #right_index=True)
            cols_to_drop = [x for x in self.buildings.columns if '_x' in x or '_y' in x ]
            self.buildings = self.buildings.drop(columns = cols_to_drop).to_crs(4326)
            self.services = pd.concat([self.Provisions[service_type]['services'] for service_type in self.service_types])
            self.services = self.services.to_crs(4326)
            
        return {"houses": eval(self.buildings.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')), 
                "services": eval(self.services.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')), 
                "provisions": {service_type: self._provision_matrix_transform(self.Provisions[service_type]['destination_matrix']) for service_type in self.service_types}}

    def _calculate_provisions(self, Provisions, service_type):
        df = pd.DataFrame.from_dict(dict(self.nx_graph.nodes(data=True)), orient='index')
        self.graph_gdf = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = self.city_crs)
        from_houses = self.graph_gdf['geometry'].sindex.nearest(Provisions['buildings']['geometry'], 
                                                                return_distance = True, 
                                                                return_all = False) 
        to_services = self.graph_gdf['geometry'].sindex.nearest(Provisions['services']['geometry'], 
                                                                return_distance = True, 
                                                                return_all = False)

        Provisions['distance_matrix'] = pd.DataFrame(0, index = to_services[0][1], 
                                                        columns = from_houses[0][1])
        
        nk_dists = nk.distance.SPSP(G = Provisions['selected_graph'], sources = Provisions['distance_matrix'].index.values).run()
        Provisions['distance_matrix'] =  Provisions['distance_matrix'].apply(lambda x: self._get_nk_distances(nk_dists,x), axis =1)
        Provisions['distance_matrix'].index = Provisions['services'].index
        Provisions['distance_matrix'].columns = Provisions['buildings'].index
        Provisions['destination_matrix'] = pd.DataFrame(0, index = Provisions['distance_matrix'].index, 
                                                           columns = Provisions['distance_matrix'].columns)
        Provisions['destination_matrix'] = self._provision_loop(Provisions['buildings'].copy(), 
                                                                Provisions['services'].copy(), 
                                                                Provisions['distance_matrix'].copy(), 
                                                                Provisions['normative_distance'], 
                                                                Provisions['destination_matrix'].copy(),
                                                                service_type )
        return Provisions        

    @staticmethod
    def _restore_user_provisions(user_provisions):
        restored_user_provisions = pd.DataFrame(user_provisions)
        restored_user_provisions = pd.DataFrame(user_provisions, columns = ['service_id','house_id','demand']).groupby(['service_id','house_id']).first().unstack()
        restored_user_provisions = restored_user_provisions.droplevel(level = 0, axis = 1)
        restored_user_provisions.index.name = None
        restored_user_provisions.columns.name = None
        restored_user_provisions = restored_user_provisions.fillna(0)

        return restored_user_provisions
    
    @staticmethod
    def _provision_matrix_transform(Provisions):
        provisions_transformed = []
        for service_id in Provisions.index:
            loc = Provisions.loc[service_id]
            loc = loc[loc > 0].to_dict().items()
            provisions_transformed.extend({'house_id': k, 
                                           'demand': v,
                                           'service_id': service_id} for k,v in loc)
        return provisions_transformed

    def recalculate_provisions(self, ):
        
        for service_type in self.service_types:
            print(service_type)
            normative_distance = self.service_types_normatives.loc[service_type].dropna().copy(deep = True)
            try:
                self.new_Provisions[service_type]['normative_distance'] = normative_distance['walking_radius_normative']
                self.new_Provisions[service_type]['selected_graph'] = self.graph_nk_length
                print('walking_radius_normative')
            except:
                self.new_Provisions[service_type]['normative_distance'] = normative_distance['public_transport_time_normative']
                self.new_Provisions[service_type]['selected_graph'] = self.graph_nk_time
                print('public_transport_time_normative')
            
            self.new_Provisions[service_type]['buildings'] = self.user_changes_buildings.copy(deep = True)
            self.new_Provisions[service_type]['services'] = self.user_changes_services[self.user_changes_services['service_code'] == service_type].copy(deep = True)

            self.new_Provisions[service_type] =  self._calculate_provisions(self.new_Provisions[service_type], service_type)
            self.new_Provisions[service_type]['buildings'], self.new_Provisions[service_type]['services'] = self._additional_options(self.new_Provisions[service_type]['buildings'].copy(), 
                                                                                                                                     self.new_Provisions[service_type]['services'].copy(),
                                                                                                                                     self.new_Provisions[service_type]['distance_matrix'].copy(),
                                                                                                                                     self.new_Provisions[service_type]['destination_matrix'].copy(),
                                                                                                                                     self.new_Provisions[service_type]['normative_distance'],
                                                                                                                                     service_type,
                                                                                                                                     self.user_selection_zone,
                                                                                                                                     self.valuation_type)
            self.new_Provisions[service_type]['buildings'], self.new_Provisions[service_type]['services'] = self._get_provisions_delta(service_type)

        cols_to_drop = [x for x in self.user_changes_buildings.columns for service_type in self.service_types if service_type in x]
        self.user_changes_buildings = self.user_changes_buildings.drop(columns = cols_to_drop)
        for service_type in self.service_types:
            self.user_changes_buildings = self.user_changes_buildings.merge(self.new_Provisions[service_type]['buildings'], 
                                                                            left_on = 'functional_object_id', 
                                                                            right_on = 'functional_object_id')
                                                                          #left_index=True, 
                                                                          #right_index=True)
        cols_to_drop = [x for x in self.user_changes_buildings.columns if '_x' in x or '_y' in x ]
        self.user_changes_buildings = self.user_changes_buildings.drop(columns = cols_to_drop).to_crs(4326)
        self.user_changes_services = pd.concat([self.new_Provisions[service_type]['services'] for service_type in self.service_types])
        self.user_changes_services = self.user_changes_services.to_crs(4326)

        return {"houses": eval(self.user_changes_buildings.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')), 
                "services": eval(self.user_changes_services.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')), 
                "provisions": {service_type: self._provision_matrix_transform(self.new_Provisions[service_type]['destination_matrix']) for service_type in self.service_types}}

    def _get_provisions_delta(self, service_type):
        #bad performance 
        #bad code
        #rewrite to df[[for col.split()[0] in ***]].sub(other[col])
        services_delta_cols = ['capacity_delta', 'capacity_left_delta', 'carried_capacity_within_delta', 'carried_capacity_without_delta']
        buildsing_delta_cols = [f'{service_type}_provison_value_delta', 
                                f'{service_type}_service_demand_left_value_{self.valuation_type}_delta', 
                                f'{service_type}_service_demand_value_{self.valuation_type}_delta',
                                f'{service_type}_supplyed_demands_within_delta',
                                f'{service_type}_supplyed_demands_without_delta']
        if self.buildings_old_values is not None:
            for col in buildsing_delta_cols:
                d = self.buildings_old_values[col.split('_delta')[0]].sub(self.new_Provisions[service_type]['buildings'][col.split('_delta')[0]], fill_value = 0)
                d = d.loc[self.new_Provisions[service_type]['buildings'].index]
                self.new_Provisions[service_type]['buildings'][col] =  d
        if self.services_old_values is not None:
            for col in services_delta_cols:
                d =  self.services_old_values[col.split('_delta')[0]].sub(self.new_Provisions[service_type]['services'][col.split('_delta')[0]], fill_value = 0) 
                d = d.loc[self.new_Provisions[service_type]['services'].index]
                self.new_Provisions[service_type]['services'][col] = d
        return self.new_Provisions[service_type]['buildings'], self.new_Provisions[service_type]['services'] 

    @staticmethod
    def _additional_options(buildings, services, Matrix, Provisions, normative_distance, service_type, selection_zone, valuation_type): 
        #clear matrix same size as buildings and services if user sent sth new
        cols_to_drop = list(set(set(Matrix.columns.values) - set(buildings.index.values)))
        rows_to_drop = list(set(set(Matrix.index.values) - set(services.index.values)))
        Matrix = Matrix.drop(index=rows_to_drop, 
                                columns=cols_to_drop, 
                                errors = 'irgonre')
        Provisions = Provisions.drop(index=rows_to_drop, 
                                    columns=cols_to_drop, 
                                    errors = 'irgonre')
                                                
        #bad performance 
        #bad code
        #rewrite to vector operations [for col in ****]
        buildings[f'{service_type}_service_demand_left_value_{valuation_type}'] = buildings[f'{service_type}_service_demand_value_{valuation_type}'] 
        buildings[f'{service_type}_supplyed_demands_within'] = 0
        buildings[f'{service_type}_supplyed_demands_without'] = 0
        services['capacity_left'] = services['capacity']
        services['carried_capacity_within'] = 0
        services['carried_capacity_without'] = 0
        for i in range(len(Provisions)):
            loc = Provisions.iloc[i]
            s = Matrix.loc[loc.name] <= normative_distance
            within = loc[s]
            without = loc[~s]
            within = within[within > 0]
            without = without[without > 0]

            buildings[f'{service_type}_service_demand_left_value_{valuation_type}'] = buildings[f'{service_type}_service_demand_left_value_{valuation_type}'].sub(within.add(without, fill_value= 0), fill_value = 0)
            buildings[f'{service_type}_supplyed_demands_within'] = buildings[f'{service_type}_supplyed_demands_within'].add(within, fill_value = 0)
            buildings[f'{service_type}_supplyed_demands_without'] = buildings[f'{service_type}_supplyed_demands_without'].add(without, fill_value = 0)
            services.at[loc.name,'capacity_left'] = services.at[loc.name,'capacity_left'] - within.add(without, fill_value= 0).sum()
            services.at[loc.name,'carried_capacity_within'] = services.at[loc.name,'carried_capacity_within'] + within.sum()
            services.at[loc.name,'carried_capacity_without'] = services.at[loc.name,'carried_capacity_without'] + without.sum()

        buildings[f'{service_type}_provison_value'] = buildings[f'{service_type}_supplyed_demands_within'] / buildings[f'{service_type}_service_demand_value_{valuation_type}']
        services['service_load'] = services['capacity'] - services['capacity_left']
        if selection_zone:
            buildings['is_shown'] = buildings.within(selection_zone)
            a = buildings['is_shown'].copy()
            t = Provisions[[c for c in Provisions.columns if c in list(a[a].index.values)]]
            t['is_shown'] = t.apply(lambda x: len(x[x > 0])>0, axis = 1)
            services['is_shown'] = t['is_shown']
        else:
            buildings['is_shown'] = True
            services['is_shown'] = True
        buildings = buildings[[x for x in buildings.columns if service_type in x] + ['is_shown','functional_object_id']]
        return buildings, services 

    def _get_nk_distances(self, nk_dists, loc):
        target_nodes = loc.index
        source_node = loc.name
        distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]

        return pd.Series(data = distances, index = target_nodes)
    
    def _declare_varables(self, loc):
        nans = loc.isna()
        index = loc.index
        name = loc.name
        data = []
        for I in index:
            if nans[I] == False:
                data.append(pulp.LpVariable(name = f"route_{name}_{I}", lowBound=0, cat = "Integer"))
            else:
                data.append(np.NaN)
        return pd.Series(index = index, data = data, name = name)

    def _provision_loop(self, houses_table, services_table, distance_matrix, selection_range, Provisions, service_type): 
        print('started')
        select = distance_matrix[distance_matrix.iloc[:] <= selection_range]
        select = select.apply(lambda x: 1/(x+1), axis = 1)
        variables = select.apply(lambda x: self._declare_varables(x), axis = 1)
        variables = variables.join(pd.DataFrame(np.NaN, 
                                                columns = list(set(select.columns) - set(select.columns)), 
                                                index = select.index))
        prob = pulp.LpProblem("problem", pulp.LpMaximize)
        for col in variables.columns:
            t = variables[col].dropna().values
            if len(t) > 0: 
                prob +=(pulp.lpSum(t) <= houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'][col],
                        f"sum_of_capacities_{col}")
            else: pass

        for index in variables.index:
            t = variables.loc[index].dropna().values
            if len(t) > 0:
                prob +=(pulp.lpSum(t) <= services_table['capacity_left'][index],
                        f"sum_of_demands_{index}")
            else:pass
        costs = []
        for index in variables.index:
            t = variables.loc[index].dropna()
            t = t * select.loc[index].dropna()
            costs.extend(t)
        prob +=(pulp.lpSum(costs),
                "Sum_of_Transporting_Costs" )
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        to_df = {}
        for var in prob.variables():
            t = var.name.split('_')
            try:
                to_df[int(t[1])].update({int(t[2]): var.value()})
            except ValueError: 
                print(t)
                pass
            except:
                to_df[int(t[1])] = {int(t[2]): var.value()}
                
        result = pd.DataFrame(to_df).transpose()
        result = result.join(pd.DataFrame(0,
                                          columns = list(set(set(Provisions.columns) - set(result.columns))),
                                          index = Provisions.index), how = 'outer')
        result = result.fillna(0)
        Provisions = Provisions + result
        axis_1 = Provisions.sum(axis = 1)
        axis_0 = Provisions.sum(axis = 0)
        services_table['capacity_left'] = services_table['capacity'].subtract(axis_1,fill_value = 0)
        houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'] = houses_table[f'{service_type}_service_demand_value_{self.valuation_type}'].subtract(axis_0,fill_value = 0)

        distance_matrix = distance_matrix.drop(index = services_table[services_table['capacity_left'] == 0].index.values,
                                        columns = houses_table[houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'] == 0].index.values,
                                        errors = 'ignore')
        print(houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'].sum(), services_table['capacity_left'].sum(),selection_range)
        selection_range += 2 * selection_range
        if len(distance_matrix.columns) > 0 and len(distance_matrix.index) > 0:
            return self._provision_loop(houses_table, services_table, distance_matrix, selection_range, Provisions, service_type)
        else: 
            return Provisions

