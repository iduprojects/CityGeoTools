from typing import Optional

import pyproj
import shapely
import geopandas as gpd
import pandas as pd
import json
import networkx as nx
import osmnx as ox
import numpy as np
import io
import pca
import momepy
import ast
import matplotlib

from matplotlib import pyplot as plt
from shapely import wkt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

matplotlib.use('Agg')


class Basics_City_Analysis_Methods():

    def __init__(self, cities_model, cities_crs):
        self.cities_inf_model = cities_model
        self.cities_crs = cities_crs

    def Geojson_Projection_Management(self, request):
        """
        :param request: json that includes FeatureCollection, and keys 'set_crs' and 'to_crs'
        :return: reprojected geojson
        """
        gdf = gpd.GeoDataFrame.from_features(request['geojson']['features'])
        gdf = gdf.set_crs(request['set_crs']).to_crs(request['to_crs'])

        return eval(gdf.to_json())

    def Request_Points_Project(self, city, request_points):
        """
        :param city: city name to get access to crs --> str
        :param request_points: list of lists containing x,y coordinates --> list
        :return: list of lists containing projected x,y coordinates --> list
        """
        city_crs = self.cities_crs[city]
        transformer = pyproj.Transformer.from_crs(4326, city_crs)

        return [transformer.transform(point[0], point[1]) for point in request_points]

    # ########################### Route_Between_Two_Points ###########################
    def Route_Between_Two_Points(self, city, travel_type, x_from, y_from, x_to, y_to, reproject=True) -> Optional[dict]:
        """
        :param city: city name to get access to information model and crs --> str
        :param travel_type: walk or drive --> str
        :param x_from, y_from: x,y coordinates of start point (in epsg:4326 if reproject == False) --> float
        :param x_to, y_to: x,y coordinated of destination point (in epsg:4326 if reproject == False) --> float
        :param reproject: if it's necessary to reproject point to epsg:4326

        :return: FeatureCollection with route line or None if path does not exists
        """
        city_inf_model, city_crs = self.cities_inf_model[city], self.cities_crs[city]
        request_points = [[float(x_from), float(y_from)], [float(x_to), float(y_to)]]
        if reproject:
            p1, p2 = self.Request_Points_Project(city, request_points)
        else:
            p1, p2 = request_points[0], request_points[1]

        graph = getattr(city_inf_model, travel_type + "_graph")
        try:
            graph_path = nx.shortest_path(graph, source=ox.distance.nearest_nodes(graph, p1[0], p1[1]),
                                          target=ox.distance.nearest_nodes(graph, p2[0], p2[1]))
        except nx.NetworkXNoPath:
            return None
        print(request_points)
        complete_route = [(graph.nodes[k]['x'], graph.nodes[k]['y']) for k in graph_path]
        route_len = float(shapely.geometry.LineString(complete_route).length)

        route_line = {"type": "FeatureCollection",
                      "features": [{'type': 'Feature',
                                    'properties': {'Name': 'Route', 'Len': route_len},
                                    'geometry': {'type': 'LineString',
                                                 'coordinates': complete_route}}]}
        return route_line

    # ###################################### Isochrone ######################################
    def walk_drive_isochrone(self, city, travel_type, x_from, y_from, weight_type, weight_value):
        """
        :param city: city name to get access to information model and crs --> str
        :param travel_type: 'walk' or 'drive' --> str
        :param x_from, y_from: x,y coordinates of start point in epsg:4326 --> float
        :param weight_type: weight type for searching shortest route ('meter' or 'time') --> str
        :param weight_value: weight value for searching shortest route --> int
        :return: FeatureCollection with isochrones geometry and type
        """
        city_inf_model = self.cities_inf_model[city]
        city_crs = self.cities_crs[city]
        point = self.Request_Points_Project(city, [[float(x_from), float(y_from)]])[0]

        graph = getattr(city_inf_model, travel_type + "_graph")
        if travel_type == 'drive':
            travel_speed = 17 * 1000 / 60
            type_ru = "Личный транспорт"
        elif travel_type == 'walk':
            travel_speed = 5 * 1000 / 60
            type_ru = "Пешком"

        graph_node = ox.distance.nearest_nodes(graph, point[0], point[1])
        for u, v, data in graph.edges(data=True):
            data['time'] = data['length'] / travel_speed

        distance_type = "length" if weight_type == "meter" else "time"
        subgraph = nx.ego_graph(graph, graph_node, radius=int(weight_value), distance=distance_type)
        node_points = [shapely.geometry.Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
        isochrones = gpd.GeoSeries(node_points).set_crs(city_crs).to_crs(4326).unary_union.convex_hull
        gdf = gpd.GeoDataFrame({"travel_type": [type_ru], "weight_type": [weight_type]},
                               geometry=[isochrones]).set_crs(4326)

        return {"isochrone": json.loads(gdf.to_json()), "routes": None, "stops": None}

    # ############################## Transport isochrone ##############################
    def transport_isochrone(self, city, travel_type, x_from, y_from, weight_value, weight_type="weight", routes=False):
        """
        :param city: city name to get access to information model and crs --> str
        :param travel_type: 'public_transport' --> str
        :param x_from, y_from: x,y coordinates of start point in epsg:4326 --> float
        :param weight_type: weight type for searching shortest route ('meter' or 'time') --> str
        :param weight_value: weight_value: weight value for searching shortest route --> int
        :return: FeatureCollection with isochrone's geometry and type
        """
        city_inf_model = self.cities_inf_model[city]
        city_crs = self.cities_crs[city]
        multi_modal_graph = city_inf_model.public_transport_graph.copy()
        start_point = self.Request_Points_Project(city, [[float(x_from), float(y_from)]])

        if weight_type == "meter":
            raise ValueError("The weight type isn't supported for public transport isochrones.")

        start_node = ox.distance.nearest_nodes(G=multi_modal_graph, X=start_point[0][0], Y=start_point[0][1])
        ego_pathes_len = nx.single_source_dijkstra_path_length(multi_modal_graph, str(start_node),
                                                               cutoff=int(weight_value), weight="weight")

        sub_nodes = gpd.GeoDataFrame(ego_pathes_len.values(), index=ego_pathes_len.keys(), columns=["path_len"])
        sub_nodes['residual_time'] = int(weight_value) - sub_nodes["path_len"]
        sub_nodes["geometry"] = list(map(lambda x: shapely.geometry.Point(eval(x)), sub_nodes.index))
        sub_nodes = sub_nodes.set_geometry("geometry").set_crs(city_crs)

        sub_sub_nodes = sub_nodes[sub_nodes['residual_time'] > 0].sort_values(by='residual_time', ascending=False)

        sub_sub_nodes['residual_time'] = sub_sub_nodes['residual_time'] * 66.66  # walk speed
        sub_sub_nodes['geometry'] = sub_sub_nodes.buffer(sub_sub_nodes['residual_time'])
        sub_sub_nodes['geometry'] = sub_sub_nodes['geometry'].apply(lambda x: x.simplify(tolerance=0.45))

        isochrone = gpd.GeoDataFrame({"travel_type": ["Общественный транспорт"], "weight_type": [weight_type]},
                                     geometry=[sub_sub_nodes['geometry'].unary_union]).set_crs(city_crs).to_crs(4326)
        isochrone = json.loads(isochrone.to_json())

        if routes:
            sub_graph = multi_modal_graph.subgraph(sub_sub_nodes.index)
            sub_graph = nx.convert_node_labels_to_integers(sub_graph)
            stops, routes = momepy.nx_to_gdf(sub_graph, points=True, lines=True, spatial_weights=False)
            routes = routes[routes["type"] != "walk"][["type", "length", "weight", "geometry"]].round(2)
            routes = routes.drop_duplicates().rename(columns={"weight": "time"})
            routes["geometry"] = routes["geometry"].apply(lambda x: wkt.loads(x))
            stops = stops[["bus", "trolleybus", "tram", "subway", "geometry"]].dropna()
            stops = stops.replace({True: "True", False: "False"})

            routes = json.loads(routes.set_crs(city_crs).to_crs(4326).to_json())
            stops = json.loads(stops.set_crs(city_crs).to_crs(4326).to_json())
        
        else: 
            routes = None
            stops = None

        return {"isochrone": isochrone, "routes": routes, "stops": stops}

    # #####################  Get services #####################
    def Get_Services(self, city, user_request):

        city_inf_model = self.cities_inf_model[city]
        services = city_inf_model.Services.copy()

        services_select = services
        if user_request["service_types"] is not None:
            service_types = user_request["service_types"]
            services_select = services_select[services_select["service_code"].isin(service_types)]

        if user_request["area"] is not None:
            area = user_request["area"]
            area_type = list(area.keys())[0]
            area_id = list(area.values())[0]
            services_select = services_select[services_select[f"{area_type}_id"] == area_id]

        return services_select.to_crs(4326).to_json()

    # ##################### Services clusterization  ######################
    def Services_Clusterization(self, city, user_request,
                                condition={"default": "default"}, n_std=2, area=None):

        def Get_Service_Cluster(services_select, condition):
            services_coords = pd.DataFrame({"x": services_select.geometry.x,
                                            "y": services_select.geometry.y})
            clusterization = linkage(services_coords.to_numpy(), method="ward")
            if list(condition.keys())[0] == "default":
                services_select["cluster"] = fcluster(clusterization, t=4000, criterion="distance")
            else:
                services_select["cluster"] = fcluster(clusterization, t=list(condition.values())[0],
                                                      criterion=list(condition.keys())[0])
            return services_select

        def Service_Groups(loc, n_std):
            if len(loc) > 1:
                X = pd.DataFrame({"x": loc.x,
                                  "y": loc.y})
                X = X.to_numpy()
                outlier = pca.spe_dmodx(X, n_std=n_std)[0]["y_bool_spe"]
                return pd.Series(data=outlier.values, index=loc.index)
            else:
                return pd.Series(data=True, index=loc.index)

        def Get_Service_Ratio(loc):
            all_services = loc["index"].count()
            services_count = loc.groupby("service_code")["index"].count()
            return (services_count / all_services).round(2)

        def Get_Cluster_Param(services_loc):
            cluster_service = services_loc.groupby(["cluster"]).apply(lambda x: Get_Service_Ratio(x))

            if isinstance(cluster_service, pd.Series):
                cluster_service = cluster_service.unstack(level=1, fill_value=0)

            cluster_param = services_loc.groupby("cluster")[["houses_in_radius", "people_in_radius",
                                                             "service_load", "loaded_capacity",
                                                             "reserve_resource"]].mean().round()
            cluster_param = cluster_param.add_prefix("mean_")
            return cluster_service, cluster_param

        city_inf_model = self.cities_inf_model[city]
        city_crs = self.cities_crs[city]
        services = city_inf_model.Services.copy()

        service_types = user_request.get("service_types")
        if user_request["area"] is not None:
            area = user_request["area"]
        if user_request["n_std"] is not None:
            n_std = user_request["n_std"]

        # Select services by types and area
        services_select = services[services["service_code"].isin(service_types)]
        if area:
            area_type = list(area.keys())[0]
            area_id = list(area.values())[0]
            services_select = services_select[services_select[f"{area_type}_id"] == area_id]

        if len(services_select) <= 1:
            return None

        # Clusterization
        services_select = Get_Service_Cluster(services_select, condition)
        # Find outliers of clusters and exclude it
        outlier = services_select.groupby("cluster")["geometry"].apply(lambda x: Service_Groups(x, n_std))

        cluster_normal = 0
        if any(~outlier):
            services_normal = services_select[~outlier]

            if len(services_normal) > 0:
                # Get normal cluster parametrs
                cluster_service, cluster_param = Get_Cluster_Param(services_normal)

                # Get MultiPoint from cluster Points and make polygon
                polygons_normal = services_normal.dissolve("cluster").convex_hull
                df_clusters_normal = pd.concat([cluster_param, cluster_service,
                                                polygons_normal.rename("geometry")],
                                               axis=1).reset_index(drop=True)
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
            cluster_service, cluster_param = Get_Cluster_Param(services_outlier)
            cluster_param = cluster_param.join(services_outlier.set_index("cluster")["geometry"])
            df_clusters_outlier = cluster_param.join(cluster_service)

        else:
            df_clusters_outlier = None

        df_clusters = pd.concat([df_clusters_normal, df_clusters_outlier]).fillna(0)
        df_clusters["geometry"] = df_clusters["geometry"].buffer(50, join_style=3)
        df_clusters = df_clusters.reset_index().rename(columns={"index": "cluster_id"})

        df_clusters = df_clusters.set_crs(city_crs).to_crs(4326)

        return df_clusters.to_json()

    # ##################### Blocks clusterization  #####################
    def Blocks_Clusterization(self, city, user_request, clusters_number="default", method="get_blocks"):

        city_inf_model = self.cities_inf_model[city]
        services = city_inf_model.Services.copy()
        blocks = city_inf_model.Base_Layer_Blocks.copy()

        user_services = user_request["service_types"]
        if "clusters_number" in user_request:
            clusters_number = user_request["clusters_number"]

        service_in_blocks = services.groupby(["block_id", "service_code"])["index"].count().unstack(fill_value=0)
        without_services = blocks["id"][~blocks["id"].isin(service_in_blocks.index)].values
        without_services = pd.DataFrame(columns=service_in_blocks.columns, index=without_services).fillna(0)
        service_in_blocks = pd.concat([without_services, service_in_blocks])

        # Select columns based on user request
        service_in_blocks = service_in_blocks[user_services]

        # Clusterization
        clusterization = linkage(service_in_blocks, method="ward")

        if method == "get_blocks":

            # Get cluster numbers. If user doesn't specified the number of clusters, use default value.
            # The default value is determined by the rate of change in the distance between clusters
            if clusters_number == "default":
                distance = clusterization[-100:, 2]
                clusters = np.arange(1, len(distance) + 1)
                acceleration = np.diff(distance, 2)[::-1]
                series_acceleration = pd.Series(acceleration, index=clusters[:-2] + 1)
                # There are always more than two clusters
                series_acceleration = series_acceleration.iloc[1:]
                clusters_number = series_acceleration.idxmax()

            service_in_blocks["cluster_labels"] = fcluster(clusterization, t=int(clusters_number), criterion="maxclust")
            blocks = blocks.join(service_in_blocks, on="id")

            # Get average number of services per cluster
            mean_services_number = service_in_blocks.groupby("cluster_labels")[user_services].mean().round()

            # Get deviations from the average for blocks
            mean_services_number = service_in_blocks[["cluster_labels"]].join(mean_services_number, on="cluster_labels")
            blocks = blocks.join(service_in_blocks[user_services] - mean_services_number[user_services], on="id",
                                 rsuffix="_deviation")

            return blocks.to_crs(4326).to_json()

        elif method == "get_dendrogram":

            img = io.BytesIO()

            # Draw dendrogram
            plt.figure(figsize=(20, 10))
            plt.title("Dendrogram")
            plt.xlabel("Distance")
            plt.ylabel("Block clusters")
            dn = dendrogram(clusterization, p=7, truncate_mode="level")
            plt.savefig(img, format="png")
            plt.close()
            img.seek(0)

            return img

    # ##################### Diversity  #####################
    def Get_Diversity(self, service_type):

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        blocks = city_inf_model.Block_Diversity.copy().dropna(subset=["mo_id"])
        mo = city_inf_model.Base_Layer_Municipalities.copy()

        blocks = blocks[[f"{service_type}_diversity", "mo_id", "geometry"]].to_crs(4326)
        mo_diversity = round(blocks.groupby(["mo_id"])[f"{service_type}_diversity"].mean(), 2)
        mo_diversity = mo[["id", "geometry"]].join(mo_diversity, on="id").to_crs(4326)

        # Pack the response
        result = {"municipalities": eval(mo_diversity.fillna("None").to_json()),
                  "blocks": dict(map(
                      lambda mo_id: (int(mo_id),
                                     eval(blocks[blocks["mo_id"] == mo_id].fillna("None").reset_index(
                                         drop=True).to_json())),
                      blocks.mo_id.unique()))}
        return result

    # ############################################ Provision ###################################################

    # TODO: Columns houses_demand and _service_provision_indices must contain indices from functional_object_id
    # TODO: Geojson services_provision must have a column 'houses_total_demand'

    def get_provision(self, service_type, area=None, provision_type="calculated", city="Saint_Petersburg",
                      is_weighted=False, service_coef=None, return_dfs=False, detailed_info=True):
        """
        :param city: city to chose data and projection --> str
        :param service_type: one of service types --> str / list of types --> list
        :param area: dict that contains area type as key and area index (or geometry) as value --> int or FeatureCollection
        :param provision_type: define provision calculation method --> str
                "calculated" - provision based on calculated demand, "normative" - provision based on normative demand

        :param is_weighted: option that define if weighted provision as result will return --> bool (default False)
                False - service provision evaluations will be returned separately for each service type (bars_provision)
                True - weighted service evaluation wil be return (provision_weighted)
        :param service_coef: if is_weighted is True, service_coef must be dictionary containing service type as key
                            and coefficient as value. Important: sum of coefficient must be equal 1
        :param: in_available_area: show only objects in available area --> bool (default False)

        :return: dict containing FeatureCollection of houses and FeatureCollection of services.
                houses properties - id, address, population, demand, number of available services,
                                    house provision of specified service type (or weighted provision)
                services properties - id, address, service type, service_name, total demand, capacity
        """
        if provision_type != "calculated" and provision_type != "normative":
            return None, "Provision type is invalid"

        city_inf_model, city_crs = self.cities_inf_model[city], self.cities_crs[city]
        houses_provision = city_inf_model.houses_provision.copy().set_crs(city_crs)
        services_provision = city_inf_model.services_provision.copy().set_crs(city_crs)

        # Get houses in area. If area is not specified, geometry of all city will be used
        # TODO: this block of code can be written as a separate function for several metrics
        if area and type(area) is dict:
            area_type = list(area.keys())[0]
            if area_type == "mo" or area_type == "district":
                houses_in_area = houses_provision[houses_provision[area_type + "_id"] == area[area_type]]
            elif area_type == "polygon":
                polygon_geojson = gpd.GeoDataFrame.from_features(area[area_type])
                polygon_geojson = polygon_geojson.set_crs(area[area_type]["crs"]["properties"]["name"]).to_crs(city_crs)
                houses_centroid = houses_provision.centroid
                houses_in_area = houses_provision[houses_centroid.within(polygon_geojson.geometry[0])]
            else:
                return None, "Area type is incorrect"
        else:
            houses_in_area = houses_provision

        # TODO: change key 'service_type' in user request to 'service_types'
        service_types = service_type
        if type(service_type) is str:
            service_types = [service_type]

        service_types_to_del = []
        for s in service_types:
            try:
                houses_provision[s + "_demand_" + provision_type]
            except KeyError:
                service_types_to_del.append(s)
        service_types = [s for s in service_types if s not in service_types_to_del]

        provision_column = [service_type + "_provision_" + provision_type for service_type in service_types]
        demand_column = [service_type + "_demand_" + provision_type for service_type in service_types]
        indices_column_original = [service_type + "_indices_original_" + provision_type for service_type in service_types]
        indices_column_processed = [service_type + "_indices_processed_" + provision_type for service_type in service_types]

        provision_df = houses_in_area[provision_column].round(2)
        demand_df = houses_in_area[demand_column]

        # Count services in house's available area
        num_available_services = houses_in_area[indices_column_original].apply(
            lambda col: col.apply(lambda row: len(row) if type(row) is list else 0))
        num_available_services.columns = num_available_services.columns.str.replace(
            "_indices_original_" + provision_type, "_num_available_services")

        houses = pd.concat([houses_in_area[["address", "population", "geometry", "block_id", "mo_id", "district_id"]],
                            provision_df, demand_df, num_available_services], axis=1)
        if detailed_info:
            houses = houses.join(self.add_columns_with_unprovided_people(
                houses_in_area, demand_df, service_types, provision_type, indices_column_original, indices_column_processed
            ))

        # Calculate weighted provision
        if is_weighted:
            houses["provision_weighted"] = self.get_weighted_provision(houses, provision_column, service_coef)
            houses = houses.drop(list(houses.filter(regex="num_available_services").columns), axis=1)

        all_services_loc = houses_in_area[indices_column_processed].apply(lambda x: x.sum(), axis=1).explode()
        all_services_loc = list(set(all_services_loc.dropna()))
        all_services_loc = services_provision[services_provision.index.isin(all_services_loc)].index
        available_service_loc = houses_in_area[indices_column_original].apply(lambda x: x.sum(), axis=1).explode()
        available_service_loc = list(set(available_service_loc.dropna()))

        services = services_provision[["address", "city_service_type", "service_name",
                                       "houses_total_demand_original_" + provision_type, "capacity", "geometry"]]
        services.columns = services.columns.str.replace("_original_" + provision_type, "")
        services = services.loc[all_services_loc]
        services["is_available"] = services.index.isin(available_service_loc)
        services["is_available"] = services["is_available"].replace({True: "True", False: "False"})

        if return_dfs:
            return {"houses": houses.to_crs(4326), "services": services.to_crs(4326)}

        return {"houses": eval(houses.reset_index().fillna("None").to_crs(4326).to_json()),
                "services": eval(services.reset_index().fillna("None").to_crs(4326).to_json())}

    def get_provision_info(self, object_type, functional_object_id, service_type, provision_type="calculated",
                           city="Saint_Petersburg", is_weighted=False, service_coef=None):
        """
        :param city: city to chose data and projection --> str
        :param object_type: house or service --> str
        :param functional_object_id: house or service id from DB --> int
        :param service_type: service type for provision evaluation --> str or list of str
        :param provision_type: provision_type: define provision calculation method --> str
                "calculated" - provision based on calculated demand, "normative" - provision based on normative demand

        :param is_weighted: option that define if weighted provision as result will return --> bool (default False)
                False - service provision evaluations will be returned separately for each service type (bars_provision)
                True - weighted service evaluation wil be return (provision_weighted)
        :param service_coef: if is_weighted is True, service_coef must be dictionary containing service type as key
                            and coefficient as value. Important: sum of coefficient must be equal 1

        :return: dict containing FeatureCollection of houses, FeatureCollection of services and FeatureCollection of isochrone.
                houses properties - id, address, population, demand, number of available services,
                                    house provision of specified service type (or weighted provision)
                services properties - id, address, service type, service_name, total demand, capacity
        """
        city_inf_model, city_crs = self.cities_inf_model[city], self.cities_crs[city]
        houses_provision = city_inf_model.houses_provision.copy().set_crs(city_crs)
        services_provision = city_inf_model.services_provision.copy().set_crs(city_crs)

        # TODO: change key 'service_type' in user request to 'service_types'
        service_types = service_type
        if type(service_type) is str:
            service_types = [service_type]

        service_types_to_del = []
        for s in service_types:
            try:
                houses_provision[s + "_demand_" + provision_type]
            except KeyError:
                service_types_to_del.append(s)
        service_types = [s for s in service_types if s not in service_types_to_del]

        provision_column = [service_type + "_provision_" + provision_type for service_type in service_types]
        demand_column = [service_type + "_demand_" + provision_type for service_type in service_types]
        indices_column_original = [service_type + "_indices_original_" + provision_type for service_type in service_types]
        indices_column_processed = [service_type + "_indices_processed_" + provision_type for service_type in service_types]

        houses_provision[provision_column] = houses_provision[provision_column].round(2)
        demand_df = houses_provision[demand_column]

        # Count services in house's available area
        num_available_services = houses_provision[indices_column_original].apply(
            lambda col: col.apply(lambda row: len(row) if type(row) is list else 0))
        num_available_services.columns = num_available_services.columns.str.replace(
            "_indices_original_" + provision_type, "_num_available_services")

        if object_type == "house":
            houses = houses_provision.loc[[functional_object_id]]
            all_services_loc = houses[indices_column_processed].apply(
                lambda x: [s for s in x.sum()
                           if services_provision["houses_demand_processed_normative"][s][functional_object_id] > 0],
                axis=1).explode()
            all_services_loc = list(set(all_services_loc.dropna()))
            available_service_loc = houses[indices_column_original].apply(
                lambda x: [s for s in x.sum()
                           if services_provision["houses_demand_original_normative"][s][functional_object_id] > 0],
                axis=1).explode()
            available_service_loc = list(set(available_service_loc.dropna()))
            services = services_provision.loc[all_services_loc]

            services["house_demand"] = services["houses_demand_processed_" + provision_type].apply(
                lambda x: x[functional_object_id])
            services = services[["address", "city_service_type", "service_name", "house_demand", "capacity", "geometry"]]
            services["is_available"] = services.index.isin(available_service_loc)
            services["is_available"] = services["is_available"].replace({True: "True", False: "False"})

            provision_num_people = self.add_columns_with_unprovided_people(
                houses, demand_df.loc[[functional_object_id]], service_types, provision_type, indices_column_original,
                indices_column_processed)
            houses = houses[["address", "population", "geometry"] + provision_column + demand_column]
            houses = houses.join(provision_num_people)

            if is_weighted:
                houses["provision_weighted"] = self.get_weighted_provision(houses, provision_column, service_coef)

            # Get isochrone
            if len(service_types) == 1:
                service_normative = city_inf_model.get_service_normative(service_types[0])
                houses = houses.to_crs(4326)
                houses_coord = list(houses.iloc[0].geometry.centroid.coords)[0]

                if service_normative[0] == "public_transport":
                    isochrone = self.transport_isochrone(
                        city, travel_type="public_transport", x_from=houses_coord[1], y_from=houses_coord[0],
                        weight_value=str(service_normative[1]))
                else:
                    isochrone = self.walk_drive_isochrone(
                        city, travel_type="walk", x_from=houses_coord[1], y_from=houses_coord[0], weight_type="meter",
                        weight_value=str(service_normative[1]))

        elif object_type == "service":
            services = services_provision.loc[[functional_object_id]]
            houses_indices = services["houses_demand_processed_" + provision_type].values[0]
            houses_indices = {k: houses_indices[k] for k, v in houses_indices.items() if v > 0}
            house_demand = pd.Series(houses_indices).rename("house_demand")
            houses = houses_provision.loc[list(houses_indices.keys())]

            provision_num_people = self.add_columns_with_unprovided_people(
                houses, demand_df.loc[houses.index], service_types, provision_type,
                indices_column_original, indices_column_processed)
            houses = houses[["address", "population", "geometry"] + provision_column + demand_column].join(house_demand)
            houses = houses.join(provision_num_people)

            available_houses = services["houses_demand_original_" + provision_type][functional_object_id].keys()
            houses["is_available"] = houses.index.isin(available_houses)
            houses["is_available"] = houses["is_available"].replace({True: "True", False: "False"})

            services = services[["address", "city_service_type", "service_name", "capacity", "geometry"]]
            houses = houses.join(num_available_services)

            # Get isochrone
            service_normative = city_inf_model.get_service_normative(service_types[0])
            services = services.to_crs(4326)
            service_coord = list(services.iloc[0].geometry.coords)[0]

            if service_normative[0] == "public_transport":
                isochrone = self.transport_isochrone(
                    city, travel_type="public_transport", x_from=service_coord[1], y_from=service_coord[0],
                    weight_value=str(service_normative[1]))["isochrone"]
            else:
                isochrone = self.walk_drive_isochrone(
                    city, travel_type="walk", x_from=service_coord[1], y_from=service_coord[0], weight_type="meter",
                    weight_value=str(service_normative[1]))["isochrone"]

        outcome_dict = {"houses": eval(houses.reset_index().fillna("None").to_crs(4326).to_json()),
                        "services": eval(services.reset_index().fillna("None").to_crs(4326).to_json())}

        if "isochrone" in locals():
            outcome_dict["isochrone"] = isochrone

        return outcome_dict

    def get_provision_aggregated(self, service_types, area_type, provision_type="calculated", is_weighted=False,
                                 service_coef=None, city="Saint_Petersburg"):

        city_inf_model, city_crs = self.cities_inf_model[city], self.cities_crs[city]
        block = city_inf_model.Blocks.copy().to_crs(city_crs)
        mo = city_inf_model.Municipalities.copy().to_crs(city_crs)
        district = city_inf_model.AdministrativeUnits.copy().to_crs(city_crs)

        provision = self.get_provision(service_type=service_types, provision_type=provision_type, detailed_info=False,
                                       is_weighted=is_weighted, service_coef=service_coef, return_dfs=True)
        houses = provision["houses"]
        houses_mean_provision = houses.groupby([f"{area_type}_id"]).mean().filter(regex="provision")
        units = eval(area_type).set_index("id").join(houses_mean_provision)
        return json.loads(units.reset_index().fillna("None").to_crs(4326).to_json())

    def get_top_provision_objects(self, service_types, area_type, area_value, provision_type="calculated",
                                  is_weighted=False, service_coef=None, city="Saint_Petersburg"):

        provision = self.get_provision(service_type=service_types, provision_type=provision_type, return_dfs=True,
                                       area={area_type: area_value}, is_weighted=is_weighted, service_coef=service_coef,
                                       detailed_info=False)
        houses = provision["houses"]

        city_inf_model, city_crs = self.cities_inf_model[city], self.cities_crs[city]
        services = self.cities_inf_model[city].services_provision.copy().set_crs(city_crs)
        services = services[(services[f"{area_type}_id"] == area_value) &
                            (services["city_service_type_code"].isin(service_types))]

        unprovided = houses.filter(regex="unprovided")
        unprovided = unprovided.join(unprovided.sum(axis=1).rename("total_unprovided"))
        unprovided = unprovided[unprovided["total_unprovided"] > 0]
        houses_top10_unprovided = unprovided.sort_values("total_unprovided", ascending=False)[:10]
        houses_top10_unprovided = pd.concat([houses[["address", "geometry"]], houses.filter(regex="provision")], axis=1
                                            ).join(houses_top10_unprovided, how="right")

        services = services.dropna(subset=[f"houses_demand_processed_{provision_type}"])
        services[f"houses_total_demand_processed_{provision_type}"] = services[
            f"houses_demand_processed_{provision_type}"].apply(lambda x: sum(x.values()) if x else None)
        service_vacancy_all = services["capacity"] - services[f"houses_total_demand_processed_{provision_type}"]
        service_vacancy_available = services["capacity"] - services[f"houses_total_demand_original_{provision_type}"]
        service_vacancy = pd.concat([service_vacancy_all.rename("total_vacancy"),
                                     service_vacancy_available.rename("vacancy_after_people_in_zone")], axis=1)
        services_top10_loaded = service_vacancy.sort_values(["total_vacancy", "vacancy_after_people_in_zone"])[:10]
        services_top10_loaded = services[["city_service_type_code", "capacity", "address", "geometry"]].join(
            services_top10_loaded, how="right")

        return {"houses": json.loads(houses_top10_unprovided.reset_index().fillna("None").to_crs(4326).to_json()),
                "services": json.loads(services_top10_loaded.reset_index().fillna("None").to_crs(4326).to_json())}

    def add_columns_with_unprovided_people(self, house_slice, demand_df, service_types, provision_type,
                                           indices_column_original, indices_column_processed):

        go_to_available = house_slice[indices_column_original].apply(
            lambda col: self.get_num_people(col, house_slice, provision_type, "available"), axis=1)
        go_to_available.columns = go_to_available.columns.str.replace(
            "_indices_original_" + provision_type, "_available_num_people")
        go_to_unavailable = house_slice[indices_column_processed].apply(
            lambda col: self.get_num_people(col, house_slice, provision_type, "unavailable"), axis=1)
        go_to_unavailable.columns = go_to_unavailable.columns.str.replace(
            "_indices_processed_" + provision_type, "_unavailable_num_people")

        unprovided = demand_df.to_numpy() - go_to_available.to_numpy()
        need_service = unprovided - go_to_unavailable.to_numpy()
        unprovided_columns_name = [service_type + "_unprovided_people" for service_type in service_types]
        need_service_columns_name = [service_type + "_need_service_num_people" for service_type in service_types]
        unprovided = pd.DataFrame(unprovided, columns=unprovided_columns_name, index=demand_df.index)
        need_service = pd.DataFrame(need_service, columns=need_service_columns_name, index=demand_df.index)

        return pd.concat([go_to_available, go_to_unavailable, unprovided, need_service], axis=1)

    @staticmethod
    def get_num_people(loc, houses_provision, provision_type, availability):
        people_go_to = houses_provision[provision_type + "_people_go_to_" + availability][loc.name]
        num_people = loc.apply(lambda x: sum([v for k, v in people_go_to.items() if k in x]))
        return num_people

    def get_weighted_provision(self, houses, provision_column, service_coef):
        """
        :param houses: DataFrame object with columns containing service provision evaluations --> DataFrame
        :param provision_column: list of columns in DataFrame containing service provision evaluations --> list
        :param service_coef: dictionary containing service type as key and coefficient as value --> dict
        :return: Series containing weighted provision
        """
        if service_coef:
            service_coef = np.array(list(dict(sorted(service_coef.items())).values()))
            provision_weighted = houses[sorted(provision_column)].apply(
                lambda x: np.nansum(x.values * service_coef), axis=1)
        else:
            provision_weighted = houses[provision_column].apply(lambda x: sum(x) / len(provision_column), axis=1)

        return provision_weighted.round(2)