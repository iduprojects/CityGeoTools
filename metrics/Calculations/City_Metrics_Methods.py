import geopandas as gpd
import shapely
import pandas as pd
import math
import json
import numpy as np
import shapely.wkt
import ast

from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler


class City_Metrics_Methods():

    def __init__(self, cities_model, cities_crs):

        self.cities_inf_model = cities_model
        self.cities_crs = cities_crs

    # ########################## Trafiics calculation #################################
    def Trafic_Area_Calculator(self, BCAM, request_area_geojson):

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        city_crs = self.cities_crs["Saint_Petersburg"]

        stops = city_inf_model.Public_Transport_Stops.copy()
        buildings = city_inf_model.Buildings.copy()

        request_area_geojson = gpd.GeoDataFrame.from_features(request_area_geojson['features'])
        request_area_geojson = request_area_geojson.set_crs(4326).to_crs(city_crs)
        living_buildings = buildings[buildings['population'] > 0]
        s = living_buildings.within(request_area_geojson['geometry'][0])
        selected_buildings = living_buildings.loc[s[s].index]

        if len(selected_buildings) == 0:
            return None

        selected_buildings['closest_stop_index'] = selected_buildings.apply(
            lambda x: stops['geometry'].distance(x['geometry']).idxmin(), axis=1)
        selected_buildings['geometry'] = selected_buildings.apply(
            lambda x: shapely.geometry.shape(
                BCAM.Route_Between_Two_Points(
                    "Saint_Petersburg", "walk", x['geometry'].centroid.coords[0][0], x['geometry'].centroid.coords[0][1],
                    stops.iloc[x['closest_stop_index']].geometry.coords[0][0],
                    stops.iloc[x['closest_stop_index']].geometry.coords[0][1],
                    reproject=False)['features'][0]['geometry']), axis=1)
        selected_buildings = selected_buildings[['population', 'building_id', 'geometry']].reset_index(drop=True)

        # 30% aprox value of Public transport users
        selected_buildings['population'] = selected_buildings['population'].apply(lambda x: int(x*0.3))
        selected_buildings['route_len'] = selected_buildings.length
        selected_buildings.rename(columns={'population': 'route_traffic'}, inplace=True)

        return eval(selected_buildings.to_crs(4326).to_json())

    # ##########################  Visibility analysis  #####################################
    def Visibility_Analysis(self, city, point, view_distance):

        city_inf_model = self.cities_inf_model[city]
        city_crs = self.cities_crs[city]
        buildings = city_inf_model.Buildings.copy()

        point_buffer = shapely.geometry.Point(point).buffer(view_distance)
        s = buildings.within(point_buffer)
        buildings_in_buffer = buildings.loc[s[s].index].reset_index(drop=True)
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

        for _0, _1 in splited_lines_gdf.groupby(level=0):
            splited_lines_list.append(_1.iloc[0]['geometry'].coords[-1])
        circuit = shapely.geometry.Polygon(splited_lines_list)
        if united_buildings:
            circuit = circuit.difference(united_buildings)

        view_zone = gpd.GeoDataFrame(geometry=[circuit]).set_crs(city_crs).to_crs(4326)

        return eval(view_zone.to_json())

    # ##########################  Weighted voronoi  #####################################
    def Weighted_Voronoi(self, request):
        def self_weight_list_calculation(start_value, iter_count):
            log_r = [start_value]
            self_weigth = []
            max_value = log_r[0] * iter_count
            for _ in range(iter_count):
                next_value = log_r[-1] + math.log(max_value/log_r[-1], 1.5)
                log_r.append(next_value)
                self_weigth.append(log_r[-1]-log_r[_])
            return self_weigth, log_r

        def vertex_checker(x_coords, y_coords, growth_rules, encounter_indexes, input_geojson):
            for i in range(len(growth_rules)):
                if not growth_rules[i]:
                    pass
                else:
                    for index in encounter_indexes:
                        if shapely.geometry.Point(x_coords[i],y_coords[i]).within(input_geojson['geometry'][index]):
                            growth_rules[i] = False
                            break
            return growth_rules

        def growth_funtion_x(x_coords, growth_rules, iteration_weight):
            growth_x = [x_coords[i - 1] + iteration_weight*math.sin(2 * math.pi * i / 65)
                        if growth_rules[i - 1] else x_coords[i - 1] for i in range(1, len(x_coords)+1)]
            return growth_x

        def growth_funtion_y(y_coords, growth_rules, iteration_weight):
            growth_y = [y_coords[i - 1] + iteration_weight*math.cos(2 * math.pi * i / 65)
                        if growth_rules[i-1] else y_coords[i-1] for i in range(1, len(y_coords)+1)]
            return growth_y

        city = request["city"]
        city_crs = self.cities_crs[city]
        json_from_flask_request = request["geojson"]
        iter_count = 300

        input_geojson = gpd.GeoDataFrame.from_features(json_from_flask_request)
        input_geojson = input_geojson.set_crs(json_from_flask_request["crs"]["properties"]["name"]).to_crs(city_crs)
        input_geojson['init_centroid'] = input_geojson.apply(lambda x: list(x['geometry'].coords)[0], axis=1)
        input_geojson['geometry'] = input_geojson.apply(
            lambda x: shapely.geometry.Polygon([
                [list(x['geometry'].coords)[0][0] + x['weight'] * math.sin(2 * math.pi * i / 65),
                 list(x['geometry'].coords)[0][1] + x['weight'] * math.cos(2 * math.pi * i / 65)]
                for i in range(1, 65)]), axis=1
        )
        input_geojson['x'] = input_geojson.apply(
            lambda x: list(list(zip(*list(x['geometry'].exterior.coords)))[0]), axis=1)
        input_geojson['y'] = input_geojson.apply(
            lambda x: list(list(zip(*list(x['geometry'].exterior.coords)))[1]), axis=1)

        input_geojson['self_weight'] = input_geojson.apply(
            lambda x: self_weight_list_calculation(x['weight'], iter_count)[0], axis=1)
        input_geojson['self_radius'] = input_geojson.apply(
            lambda x: self_weight_list_calculation(x['weight'], iter_count)[1], axis=1)
        input_geojson['vertex_growth_allow_rule'] = input_geojson.apply(
            lambda x: [True for x in range(len(x['x']))], axis=1)

        temp = pd.DataFrame({'x': input_geojson.apply(
            lambda x: growth_funtion_x(x['x'], x['vertex_growth_allow_rule'], x['self_radius'][-1]), axis=1),
                            'y': input_geojson.apply(
            lambda x: growth_funtion_y(x['y'], x['vertex_growth_allow_rule'], x['self_radius'][-1]), axis=1)}).apply(
            lambda x: shapely.geometry.Polygon(tuple(zip(x['x'], x['y']))), axis=1)

        input_geojson['encounter_rule_index'] = [[y for y in range(len(temp))
                                                  if y != x if temp[x].intersects(temp[y])] for x in range(len(temp))]
        for i in range(iter_count):
            input_geojson['x'] = input_geojson.apply(
                lambda x: growth_funtion_x(x['x'], x['vertex_growth_allow_rule'], x['self_weight'][i]), axis=1)
            input_geojson['y'] = input_geojson.apply(
                lambda x: growth_funtion_y(x['y'], x['vertex_growth_allow_rule'], x['self_weight'][i]), axis=1)
            input_geojson['geometry'] = input_geojson.apply(
                lambda x: shapely.geometry.Polygon(tuple(zip(x['x'], x['y']))), axis=1)
            input_geojson['vertex_growth_allow_rule'] = input_geojson.apply(
                lambda x: vertex_checker(x['x'], x['y'], x['vertex_growth_allow_rule'], x['encounter_rule_index'],
                                         input_geojson), axis=1)

        start_points = gpd.GeoDataFrame.from_features(json_from_flask_request)
        start_points = start_points.set_crs(json_from_flask_request["crs"]["properties"]["name"]).to_crs(city_crs)
        x = [list(p.coords)[0][0] for p in start_points['geometry']]
        y = [list(p.coords)[0][1] for p in start_points['geometry']]

        centroid = shapely.geometry.Point((sum(x) / len(start_points['geometry']),
                                           sum(y) / len(start_points['geometry'])))

        buffer_untouch = centroid.buffer(start_points.distance(shapely.geometry.Point(centroid)).max() * 1.4)
        buffer_untouch = gpd.GeoDataFrame(data={'id': [1]}, geometry=[buffer_untouch]).set_crs(city_crs)

        result = gpd.overlay(buffer_untouch, input_geojson, how='difference')
        input_geojson = input_geojson.to_crs(4326)
        result = result.to_crs(4326)

        return {'voronoi_polygons': eval(input_geojson[['weight', 'geometry']].to_json()),
                'deficit_zones': eval(result.to_json())}

    # #################################  Spacematrix  ######################################
    def Get_Polygon(self, mo, districts, arg, crs):

        if "polygon" in arg:
            coord = ast.literal_eval(arg.get("polygon"))
            polygon = gpd.GeoSeries(Polygon(coord)).set_crs(4326).to_crs(crs)[0]

        elif "municipalities" in arg:
            value = str(arg.get("municipalities"))
            polygon = mo[mo["name"] == value]
            polygon = polygon["geometry"].values[0]

        elif "districts" in arg:
            value = str(arg.get("districts"))
            polygon = districts[districts["name"] == value]
            polygon = polygon["geometry"].values[0]

        return polygon

    def Get_Objects(self, arg):

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        city_crs = self.cities_crs["Saint_Petersburg"]

        buildings = city_inf_model.Spacematrix_Buildings.copy()
        blocks = city_inf_model.Spacematrix_Blocks.copy()
        mo = city_inf_model.Base_Layer_Municipalities.copy()
        districts = city_inf_model.Base_Layer_Districts.copy()

        polygon = self.Get_Polygon(mo, districts, arg, city_crs)
        b_blocks = blocks.centroid.apply(lambda x: x.within(polygon))
        blocks_within = blocks[blocks.index.isin(b_blocks[b_blocks].index)]
        buildings_within = buildings[buildings["block"].isin(blocks_within["ID"].values)]

        blocks_result = blocks_within[["area", "geometry"]].to_crs(4326).to_json()
        buildings_result = buildings_within[["type", "floors", "area", "living_space", "geometry"]].to_crs(4326).to_json()

        result = (blocks_result, buildings_result)
        return result

    def Get_Indices(self, arg):

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        city_crs = self.cities_crs["Saint_Petersburg"]
        blocks = city_inf_model.Spacematrix_Blocks.copy()
        mo = city_inf_model.Base_Layer_Municipalities.copy()
        districts = city_inf_model.Base_Layer_Districts.copy()

        ind = str(arg.get("index"))
        polygon = self.Get_Polygon(mo, districts, arg, city_crs)
        b_blocks = blocks.centroid.apply(lambda x: x.within(polygon))
        blocks_within = blocks[blocks.index.isin(b_blocks[b_blocks].index)].to_crs(4326)

        if ind == "Spacematrix":
            result = blocks_within[["cluster", ind, "FSI", "GSI", "L", "OSR", "MXI",
                                    "geometry"]].astype({"cluster": "Int8"}).to_json()
        else:
            result = blocks_within[[ind, "geometry"]].to_json()
        return result

    # ######################################### Wellbeing ##############################################
    def get_wellbeing(self, BCAM, living_situation_id=None, user_service_types=None, area=None,
                      provision_type="calculated", city="Saint_Petersburg", wellbeing_option=None, return_dfs=False):
        """
        :param BCAM: class containing get_provision function --> class
        :param living_situation_id: living situation id from DB --> int (default None)
        :param city: city to chose data and projection --> str
        :param area: dict that contains area type as key and area index (or geometry) as value --> int or FeatureCollection
        :param user_service_types: use to define own set of services and their coefficient --> dict (default None)
                with service types as keys and coefficients as values
        :param wellbeing_option: option that define which houses are viewed on map --> list of int (default None)
                                given list include left and right boundary
        :param provision_type: define provision calculation method --> str
                "calculated" - provision based on calculated demand, "normative" - provision based on normative demand

        :return: dict containing FeatureCollection of houses and FeatureCollection of services.
                houses properties - id, address, population, wellbeing evaluation
                services properties - id, address, service type, service_name, total demand, capacity
        """

        # Get service coefficient from DB or user request
        service_coef = self.parse_service_coefficients(user_service_types, living_situation_id)
        if type(service_coef) is tuple:
            return service_coef

        provision = BCAM.get_provision(list(service_coef["service_code"]), area, provision_type)
        if type(provision) is tuple:
            return provision

        houses = gpd.GeoDataFrame.from_features(provision["houses"]).set_crs(4326)
        services = gpd.GeoDataFrame.from_features(provision["services"]).set_crs(4326)
        provision_columns = houses.filter(regex="provision").replace("None", np.nan)
        unprovided_columns = list(houses.filter(regex="unprovided").columns)

        available_service_type = [t.split("_provision")[0] for t in provision_columns.columns]
        service_coef = service_coef[service_coef["service_code"].isin(available_service_type)]
        provision_columns = provision_columns.reindex(sorted(provision_columns.columns), axis=1)
        weighted_provision_columns = provision_columns * service_coef.set_axis(provision_columns.columns)["evaluation"]
        houses["mean_provision"] = weighted_provision_columns.apply(
            lambda x: x.mean() if len(x[x.notna()]) > 0 else None, axis=1)
        wellbeing = self.calculate_wellbeing(provision_columns, service_coef)
        houses = houses.drop(list(houses.filter(regex="demand").columns) + unprovided_columns +
                             list(houses.filter(regex="available").columns), axis=1).join(wellbeing)

        if wellbeing_option:
            houses = houses[houses["wellbeing"].between(*wellbeing_option)]
            # PLUG!!! There must be slice by functional object id for services

        if return_dfs:
            return {"houses": houses.to_crs(4326), "services": services.to_crs(4326)}

        return {"houses": eval(houses.reset_index().fillna("None").to_crs(4326).to_json()),
                "services": eval(services.reset_index().fillna("None").to_crs(4326).to_json())}

    def get_wellbeing_info(self, BCAM, object_type, functional_object_id, provision_type="calculated",
                           living_situation_id=None, user_service_types=None, city="Saint_Petersburg"):
        """
        :param BCAM: class containing get_provision function --> class
        :param object_type: house or service --> str
        :param functional_object_id: house or service id from DB --> int
        :param provision_type: provision_type: define provision calculation method --> str
                "calculated" - provision based on calculated demand, "normative" - provision based on normative demand
        :param living_situation_id: living situation id from DB --> int (default None)
        :param user_service_types: use to define own set of services and their coefficient --> dict (default None)
                with service types as keys and coefficients as values
        :param city: city to chose data and projection --> str

        :return: dict containing FeatureCollections of houses, services,
                service_types (only when object_type is house) and isochrone (only when object_type is service)
        """

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        # Get service coefficient from DB or user request
        service_coef = self.parse_service_coefficients(user_service_types, living_situation_id)
        if type(service_coef) is tuple:
            return service_coef

        objects = BCAM.get_provision_info(object_type, functional_object_id,
                                          list(service_coef["service_code"]), provision_type)
        if type(objects) is tuple:
            return objects
        
        houses = gpd.GeoDataFrame.from_features(objects["houses"]).fillna(-1).set_crs(4326)
        services = gpd.GeoDataFrame.from_features(objects["services"]).fillna(-1).set_crs(4326)

        provision_columns = houses.filter(regex="provision").replace("None", np.nan)
        set_demand_columns = list(houses.filter(regex="demand").columns)
        set_num_service_columns = list(houses.filter(regex="available").columns)
        unprovided_columns = list(houses.filter(regex="unprovided").columns)

        available_service_type = [t.split("_provision")[0] for t in provision_columns.columns]
        service_coef = service_coef[service_coef["service_code"].isin(available_service_type)].sort_values(
            "service_code")
        wellbeing = self.calculate_wellbeing(provision_columns, service_coef)

        if object_type == "house":
            provision_columns = provision_columns.reindex(sorted(provision_columns.columns), axis=1)
            weighted_provision_columns = provision_columns * service_coef.set_axis(provision_columns.columns)[
                "evaluation"]
            houses["mean_provision"] = weighted_provision_columns.apply(
                lambda x: x.mean() if len(x[x.notna()]) > 0 else None, axis=1)
            houses = houses.drop(set_demand_columns + set_num_service_columns + unprovided_columns, axis=1).join(
                wellbeing)
            service_types_info = self.calculate_wellbeing(provision_columns.iloc[0], service_coef, get_provision=True)

        elif object_type == "service":
            service_type = services.iloc[0]["city_service_type"]
            service_code = city_inf_model.get_service_code(service_type)
            drop_col = [col for col in set_demand_columns if service_code not in col] + \
                       [col for col in set_num_service_columns if service_code not in col]
            houses = houses.drop(drop_col + unprovided_columns, axis=1).join(wellbeing)
            isochrone = gpd.GeoDataFrame.from_features(objects["isochrone"]).set_crs(4326)

        outcome_dict = {"houses": eval(houses.reset_index(drop=True).fillna("None").to_crs(4326).to_json()),
                        "services": eval(services.reset_index(drop=True).fillna("None").to_crs(4326).to_json())}

        if "service_types_info" in locals():
            outcome_dict["service_types"] = eval(service_types_info.to_json())
        elif "isochrone" in locals():
            outcome_dict["isochrone"] = eval(isochrone.to_json())
        return outcome_dict

    def get_wellbeing_aggregated(self, BCAM, area_type, living_situation_id=None, user_service_types=None,
                                 provision_type="calculated", city="Saint_Petersburg"):

        city_inf_model, city_crs = self.cities_inf_model[city], self.cities_crs[city]
        block = city_inf_model.Base_Layer_Blocks.copy().to_crs(city_crs)
        mo = city_inf_model.Base_Layer_Municipalities.copy().to_crs(city_crs)
        district = city_inf_model.Base_Layer_Districts.copy().to_crs(city_crs)

        wellbeing = self.get_wellbeing(BCAM=BCAM, living_situation_id=living_situation_id, return_dfs=True,
                                       user_service_types=user_service_types, provision_type=provision_type)
        houses = wellbeing["houses"]
        houses_mean_provision = houses.groupby([f"{area_type}_id"]).mean().filter(regex="provision")
        houses_mean_wellbeing = houses.groupby([f"{area_type}_id"]).mean().filter(regex="wellbeing")
        houses_mean_stat = pd.concat([houses_mean_provision, houses_mean_wellbeing], axis=1)
        units = eval(area_type).set_index("id").drop(["center"], axis=1).join(houses_mean_stat)
        return json.loads(units.reset_index().fillna("None").to_crs(4326).to_json())

    def calculate_wellbeing(self, loc, coef_df, get_provision=False):

        if get_provision:
            provision = loc.sort_index()
            provision.index = [idx.split("_provision_")[0] for idx in provision.index]
            available_type = provision.notna()
            provision = provision[available_type]
            coef_df = coef_df.sort_values(by="service_code").set_index("service_code")["evaluation"][available_type]
            coef = list(coef_df)
            provision = list(provision)
            weighted_provision = [1 + 2 * coef[i] * (-1 + provision[i]) if coef[i] <= 0.5
                                  else provision[i] ** (8 * coef[i] - 3) for i in range(len(provision))]
            result = pd.DataFrame({"service_code": list(coef_df.index), "provision": provision,
                                   "coefficient": coef, "wellbeing": weighted_provision}).round(2)
            return result

        else:
            provision = loc.reindex(sorted(loc.columns), axis=1).to_numpy()
            coef_df = coef_df.sort_values(by="service_code").set_index("service_code")["evaluation"]
            coef = list(coef_df)
            weighted_provision = [list(1 + 2 * coef[i] * (-1 + provision[:, i])) if coef[i] <= 0.5
                                  else list(provision[:, i] ** (8 * coef[i] - 3)) for i in range(len(coef))]
            weighted_provision = np.array(weighted_provision).T
            general_wellbeing = np.nansum(weighted_provision * coef / sum(coef), axis=1)
            weighted_provision = np.c_[weighted_provision, general_wellbeing]
            weighted_index = [t + "_wellbeing" for t in coef_df.index] + ["wellbeing"]
            weighted_series = pd.DataFrame(weighted_provision, columns=weighted_index, index=loc.index).round(2)
            return weighted_series

    def parse_service_coefficients(self, user_service_types=None, living_situation_id=None):
        """
        :param user_service_types: use to define own set of services and their coefficient --> dict (default None)
                with service types as keys and coefficients as values
        :param living_situation_id: living situation id from DB --> int (default None)
        :return: DataFrame object containing columns with service types and coefficients --> DataFrame
        """
        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        if user_service_types and type(user_service_types) is dict:
            service_coef = pd.DataFrame([[key, user_service_types[key]] for key in user_service_types],
                                        columns=["service_code", "evaluation"])

        elif living_situation_id and (type(living_situation_id) is int or type(living_situation_id) is str):
            service_coef = city_inf_model.get_living_situation_evaluation(living_situation_id)
            if len(service_coef) == 0:
                return None, "Living situation id absents in DB"
        else:
            return None, "Invalid data to calculate well-being. Specify living situation or service types"

        # Because provision for house as service is not calculated
        if "houses" in service_coef["service_code"]:
            service_coef = service_coef[service_coef["service_code"] != "houses"]

        return service_coef






