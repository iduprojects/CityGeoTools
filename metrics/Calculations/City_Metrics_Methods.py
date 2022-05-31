import geopandas as gpd
import shapely
import pandas as pd
import math
import requests
import numpy as np
import shapely.wkt
import ast

from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler


class City_Metrics_Methods():

    def __init__(self, cities_model, cities_crs):

        self.cities_inf_model = cities_model
        self.cities_crs = cities_crs

    # ########################## Connectivity calculations ################################
    def Connectivity_Viewer(self, selected_area_geojson):

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        city_crs = self.cities_crs["Saint_Petersburg"]

        points_layer = city_inf_model.Connectivity_Metrics_Data_Points.copy()
        blocks_layer = city_inf_model.Base_Layer_Blocks.copy()

        selected_area_geojson = gpd.GeoDataFrame.from_features(selected_area_geojson['features']).set_crs(4326)
        selected_area_geojson = selected_area_geojson.set_crs(4326).to_crs(city_crs)
        blocks_layer['selection_rule'] = blocks_layer.apply(
            lambda x: selected_area_geojson['geometry'][0].contains(x['geometry']), axis=1)
        points_layer['selection_rule'] = points_layer.apply(
            lambda x: selected_area_geojson['geometry'][0].contains(x['geometry']), axis=1)
        blocks_selected = blocks_layer.loc[blocks_layer['selection_rule'] == True]
        points_selected = points_layer.loc[points_layer['selection_rule'] == True]
        blocks_selected = blocks_selected.drop(['selection_rule'], axis=1)
        points_selected = points_selected.drop(['selection_rule'], axis=1)
        selected_area_geojson['connectivity_index'] = points_selected['ratio'].mean()

        return {"selected_points": eval(blocks_selected.to_crs(4326).fillna('0').to_json()),
                "selected_blocks": eval(points_selected.fillna('0').to_crs(4326).to_json()),
                "input_area_with_connectibity_index": eval(selected_area_geojson.to_crs(4326).fillna("None").to_json())}

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

    # ##########################  Instagram  #####################################
    def get_instagram_data(self, year, season, day_time):
        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        file = city_inf_model.get_instagram_data(year, season, day_time)
        result = eval(gpd.GeoDataFrame.from_features(file).set_crs(3857).to_crs("EPSG:4326").to_json())
        return result

    # ########################## House location  #####################################
    def Get_Delta(self, block, services, input_population, service_preference, sum_delta=True):

        block = pd.Series(block).rename("ratio")
        block.index = block.index.astype("int")

        # Get service load per house
        block_ratio = pd.concat([block, services["service_code"]], axis=1, join="inner")
        load_per_house = input_population / block_ratio.groupby("service_code").sum()
        load_per_house = load_per_house.rename(columns={"ratio": "load_per_house"})
        block_ratio = block_ratio.join(load_per_house, on="service_code")
        block_ratio["load_per_house"] = block_ratio["load_per_house"] * block_ratio["ratio"]

        # Get the difference between initial service load and new service load (plus load from new house)
        add_load = block_ratio["load_per_house"]
        load = services["service_load"][block_ratio.index]
        normative = services["normative"][block_ratio.index]
        max_load = services["max_load"][block_ratio.index]
        reserve_resource = services["reserve_resource"][block_ratio.index]

        delta = reserve_resource - (max_load - round((add_load + load) * normative / 1000))
        delta_code = pd.concat([delta.rename("delta"), services["service_code"][block_ratio.index]], axis=1)

        # Select services by preference
        if not sum_delta:
            code_bool = delta_code["service_code"].isin(service_preference)
            return delta_code["delta"][code_bool]

        delta_code = delta_code.groupby("service_code")["delta"].sum()
        code_bool = delta_code.index.isin(service_preference)
        return sum(delta_code[code_bool])

    def House_Location(self, user_request):

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]

        houses = city_inf_model.Living_Buildings_Provision.copy()
        blocks = city_inf_model.Base_Layer_Blocks.copy()
        blocks = blocks.drop(blocks.filter(regex="diversity").columns, axis=1).copy()
        mo = city_inf_model.Base_Layer_Municipalities.copy()
        services = city_inf_model.Services.copy()
        ratio = city_inf_model.Hose_Location_service_ratio.copy()

        # Get user request
        input_area = float(user_request["area"])
        input_floors = float(user_request["floors"])
        input_population = int(user_request["population"])
        service_preference = user_request["service_types"]

        # Check block area
        check_area = blocks["max_parcel_area"] < input_area * 1.1
        available_blocks = blocks.drop(check_area[check_area].index).reset_index(drop="True")

        # Get ratio table. Ratio = service availability area in the block / block area
        service_ratio_bool = ratio.index.isin(available_blocks["id"])
        service_ratio_blocks = ratio[service_ratio_bool]

        services_resource_delta = service_ratio_blocks["ratio"].apply(
            lambda x: self.Get_Delta(x, services, input_population, service_preference))

        available_blocks = available_blocks.join(services_resource_delta.rename("service_resourse_minus"),
                                                 on="id").dropna(subset=["service_resourse_minus"])
        # Get sum of house resources in the block and scale it
        houses_resources = houses.filter(regex="_resource", axis=1)
        service_preference = list(map(lambda x: x + "_resource", service_preference))
        houses_resources = houses_resources[service_preference]
        scaler = MinMaxScaler((-100, 100))
        houses["mean_resources"] = np.mean(scaler.fit_transform(houses_resources.to_numpy()), axis=1)
        mean_houses_resource = houses.groupby("block_id")["mean_resources"].mean().rename("mean_houses_resource")
        available_blocks = available_blocks.join(mean_houses_resource, on="id", how="inner")
        # Get scaled scores
        scaler = MinMaxScaler((-100, 0))
        available_blocks["resourse_minus_score"] = scaler.fit_transform(
            available_blocks["service_resourse_minus"].to_numpy().reshape(-1, 1))
        scaler = MinMaxScaler((0, 100))
        available_blocks["max_parcel_area_score"] = scaler.fit_transform(
            available_blocks["max_parcel_area"].to_numpy().reshape(-1, 1))
        available_blocks = available_blocks[available_blocks["max_parcel_area_score"] > 1]
        scaler = MinMaxScaler((0, 100))
        recources = available_blocks["mean_houses_resource"].where(available_blocks["mean_houses_resource"] > 0, 0)
        available_blocks["houses_resource_score"] = scaler.fit_transform(recources.to_numpy().reshape(-1, 1))

        # Get range for weighted sum

        k = np.log2(input_area) / 3
        m = 3 + 10000 / (input_area * input_floors)

        # Get sum scores of block, scale and sort it (final scale / rank)
        available_blocks["score"] = m * available_blocks["houses_resource_score"] + \
                                    m * available_blocks["resourse_minus_score"] + \
                                    k * available_blocks["max_parcel_area_score"]
        available_blocks["score"] = scaler.fit_transform(available_blocks["score"].to_numpy().reshape(-1, 1))
        col = ["id", "mean_houses_resource", "service_resourse_minus", "score"]
        blocks = blocks.merge(available_blocks[col], how="left", left_on="id", right_on='id')

        blocks["score"] = round(blocks["score"].fillna(0))
        cols = blocks.columns.tolist()
        # cols = cols[:4] + cols[6:9] + cols[5:6] + cols[-1:]
        blocks = blocks[cols].sort_values(by="score", ascending=False).reset_index(drop=True).to_crs(4326)
        blocks = blocks.dropna(subset=["mo_id"])
        # Get mo score based on sum of blocks score, scale and sort it
        mo_score = blocks.groupby("mo_id")["score"].sum()
        mo_features = round(blocks.groupby("mo_id")["mean_houses_resource", "service_resourse_minus",
                                                    "max_parcel_area"].mean())
        mo = mo.set_index("id").join([mo_features, mo_score])
        scaler = MinMaxScaler((0, 100))
        mo["score"] = np.round(scaler.fit_transform(mo["score"].to_numpy().reshape(-1, 1)))
        mo = mo.sort_values(by="score", ascending=False).reset_index().to_crs(4326)
        # Pack the response
        result = {"municipalities": eval(mo.fillna("None").to_json()),
                  "blocks": dict(map(
                      lambda mo_id: (int(mo_id),
                                     eval(blocks[blocks["mo_id"] == mo_id].fillna("None").reset_index(drop=True).to_json())),
                      blocks.mo_id.unique()))}
        return result

    def Block_Provision(self, user_request):

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        houses = city_inf_model.Living_Buildings_Provision.copy()
        blocks = city_inf_model.Base_Layer_Blocks.copy()
        services = city_inf_model.Services.copy()
        service_ratio = city_inf_model.Hose_Location_service_ratio.copy()

        block_id = float(user_request["block_id"])
        if block_id not in service_ratio.index:
            return None

        input_population = int(user_request["population"])
        service_preference = user_request["service_types"]

        # Get average provision before house construction
        houses = houses[houses["block_id"] == block_id]
        houses_resources = houses.filter(regex="_provision", axis=1)
        service_preference_provision = list(map(lambda x: x + "_provision", service_preference))
        house_in_block = houses_resources[service_preference_provision]
        houses_provision = house_in_block.mean()
        block = blocks[blocks["id"] == block_id].reset_index(drop=True)
        provision_before = block.join(round(houses_provision.to_frame().T))
        provision_before = provision_before.fillna("None")

        # Get services available for block
        ratio = service_ratio["ratio"][block_id]
        ratio = pd.Series(ratio).rename("ratio")
        ratio.index = ratio.index.astype("int")
        services_area_code = pd.concat([ratio,
                                        services["service_code"],
                                        services["reserve_resource"],
                                        services["people_in_radius"]], join="inner", axis=1)

        # Get house resource
        services_resource_delta = self.Get_Delta(ratio, services, input_population, service_preference, sum_delta=False)
        services_area_code = services_area_code.join(services_resource_delta).dropna()
        services_area_code["reserve_resource"] = services_area_code["reserve_resource"] - services_area_code["delta"]
        services_area_code["people_in_radius"] = services_area_code["people_in_radius"] + \
                                                 (input_population * services_area_code["ratio"])
        services_area_code["resource_service"] = (services_area_code["reserve_resource"] * input_population *
                                                  services_area_code["ratio"] / services_area_code["people_in_radius"])

        house_resource = services_area_code.groupby("service_code")["resource_service"].sum()
        service_normative = services.groupby("service_code")["normative"].first().loc[service_preference]
        house_resource = pd.concat([service_normative[~service_normative.index.isin(house_resource.index)] * (-1),
                                    house_resource])

        # Get average provision after house construction
        house_evaluation = services[["service_code", "house_evaluation"]].groupby("service_code")["house_evaluation"].first()
        house_evaluation = house_evaluation.loc[house_resource.index]
        house_evaluation = house_evaluation.apply(lambda x: list(map(int, x.split(",")))).tolist()
        bool_matrix = np.array(house_evaluation) > house_resource.values.reshape(-1, 1)
        evaluation = map(lambda values: values.index(True) if True in values else 10, bool_matrix.tolist())
        house_provision = pd.Series(list(evaluation), index=house_resource.index, name="provision")
        house_provision = house_provision.add_suffix("_provision")
        new_house_provision = (house_in_block.sum() + house_provision) / (house_in_block.count() + 1)
        provision_after = block.join(round(new_house_provision.to_frame().T))
        provision_after = provision_after.fillna("None")

        result = {"provision_before": eval(provision_before.to_crs(4326).to_json()),
                  "provision_after": eval(provision_after.to_crs(4326).to_json())}

        return result

    # ###################################### House selection ##################################################
    def House_selection(self, user_request):

        def calculate_municipalities_score(municipalities_loc, houses_provosion):
            a = houses_provosion.within(municipalities_loc['geometry'])
            municipalities_score = houses_provosion.iloc[a[a].index]['score'].mean()
            provision_municipalite_association_df = pd.DataFrame(index=a[a].index, data=municipalities_loc['name'],
                                                                 columns=['municipalities_name'])

            return municipalities_score, provision_municipalite_association_df

        def calculate_user_house_score(houses_provosion_loc, social_groups_significance):
            house_score = np.sum(social_groups_significance['significance'] * houses_provosion_loc.values)

            return house_score

        def create_significance_df(social_groups_significance, user_request, default_services):

            services_to_check = default_services + user_request['user_selected_significant_services']
            for service_name in services_to_check:
                social_groups_significance.append({'code': service_types[service_name],
                                                   'service_type': service_name, "significance":1})

            services_to_check = user_request['user_selected_unsignificant_services']
            for service_name in services_to_check:
                social_groups_significance.append({'code':service_types[service_name],
                                                   'service_type': service_name, "significance":-1})

            # summirize user_send significance with existed (if exists)
            social_groups_significance = pd.DataFrame(social_groups_significance)
            social_groups_significance['significance'] = pd.DataFrame(social_groups_significance)\
                                                        .groupby(['code'])['significance']\
                                                        .transform('sum')
            try:
                # if social_groups_significance is empty
                social_groups_significance.drop(columns=['id'], inplace=True)
            except: pass

            social_groups_significance.drop_duplicates(subset=['code'], keep='first', inplace=True)
            social_groups_significance.reset_index(drop=True, inplace=True)

            return social_groups_significance

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        city_crs = self.cities_crs["Saint_Petersburg"]
        houses_provosion = city_inf_model.Living_Buildings_Provision.copy()
        municipalities = city_inf_model.Base_Layer_Municipalities.copy()
        service_types = city_inf_model.Service_types.copy()

        default_services = ['Школа', 'Детский сад']
        base_columns_houses = ['municipalities_name', 'floors', ' yearConstruct',
                               'livingSpace', 'garbageChute', 'lift',
                               'gascentral', 'hotwater', 'electricity',
                               'failure', 'population', 'avg_m2_price_rent',
                               'avg_m2_square_rent', 'min_price_rent', 'max_price_rent',
                               'avg_m2_price_sale', 'avg_m2_square_sale', 'min_price_sale',
                               'max_price_sale', 'mean_price_sale', 'mean_price_rent', 'geometry', 'score']
        scaler = MinMaxScaler(feature_range=(1, 100))

        if 'user_social_group_selection' in user_request:
            # build an series-like social group services signigicance
            social_groups_significance = city_inf_model.get_social_groups_significance(user_request)
            social_groups_significance = create_significance_df(social_groups_significance, user_request, default_services)

        elif len(user_request['user_selected_significant_services']) > 0:
            social_groups_significance = create_significance_df([], user_request, default_services)
        elif len(user_request['user_selected_significant_services']) == 0:
            return {'error': 'invalid_request'}

        social_groups_significance['code'] = social_groups_significance['code'] + '_provision'

        temp = houses_provosion[houses_provosion['mean_price_rent'].between(user_request['user_price_preferences'][0],
                                                                            user_request['user_price_preferences'][1],
                                                                            inclusive=False)]
        temp = temp.loc[:, temp.columns.isin([*social_groups_significance['code']])]

        # drop unmached services provisions
        social_groups_significance = social_groups_significance[
            social_groups_significance['code'].isin(temp.columns)].reset_index(drop=True)

        houses_provosion['score'] = temp.apply(
            lambda x: calculate_user_house_score(x, social_groups_significance), axis=1)
        houses_provosion['score'] = houses_provosion['score'].fillna(0)
        houses_provosion['score'] = scaler.fit_transform(houses_provosion['score'].values.reshape(-1, 1))

        result = list(municipalities.apply(lambda x: calculate_municipalities_score(x, houses_provosion), axis=1))
        result_houses = pd.merge(pd.concat([_[1] for _ in result]), houses_provosion, left_index=True, right_index=True)
        result_houses = result_houses.loc[:, result_houses.columns.isin([*social_groups_significance['code']] +
                                                                        base_columns_houses)]
        result_houses = gpd.GeoDataFrame(result_houses, geometry=result_houses['geometry'], crs=city_crs).to_crs(4326)
        municipalities['score'] = scaler.fit_transform(np.array([_[0] for _ in result]).reshape(-1, 1))

        response_json = {'municipalities': eval(municipalities.to_crs(4326).fillna(0).to_json()),
                         'houses': {municipalities_name: eval(result_houses[result_houses['municipalities_name'] ==
                                                                            municipalities_name].\
                                                              reset_index(drop=True).fillna('None').to_json())
                                    for municipalities_name in pd.unique(result_houses['municipalities_name'])}}
        return response_json

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

#    ############################### Service location ####################################
    def Service_location(self, user_request):

        def service_score_properties(subset, user_service_choice, Service_types):

            mean_unit_provision = subset[Service_types[user_service_choice] + '_provision'].mean()
            total_unit_resource = subset[Service_types[user_service_choice] + '_resource'].sum()
            return mean_unit_provision, total_unit_resource

        def scaler(x, min_x,max_x, scaler_range):

            scaled_x = scaler_range['min'] + ((x - min_x)*(scaler_range['max']-scaler_range['min']))/(max_x-min_x)
            return scaled_x

        def invert_scaler(x, min_x,max_x,scaler_range):

            invert_scaled_x = scaler_range['min'] + ((max_x - x)*(scaler_range['max']-scaler_range['min']))/(max_x-min_x)
            return invert_scaled_x

        city_inf_model = self.cities_inf_model["Saint_Petersburg"]
        Service_types = city_inf_model.Service_types.copy()
        Base_Layer_Municipalities = city_inf_model.Base_Layer_Municipalities.copy()
        Base_Layer_Blocks = city_inf_model.Base_Layer_Blocks.copy()
        Living_Buildings_Provision = city_inf_model.Living_Buildings_Provision.copy()
        Commercial_rent_ads = city_inf_model.Commercial_rent_ads.copy()

        choice_subset = Living_Buildings_Provision[[Service_types[user_request['user_service_choice']] + '_provision',
                                                    Service_types[user_request['user_service_choice']] + '_resource',
                                                    'block_id',
                                                    'mo_id']]
        # Blocks
        Base_Layer_Blocks[['mean_unit_provision',
                           'total_unit_resource']] = Base_Layer_Blocks.apply(
            lambda x: service_score_properties(choice_subset[choice_subset['block_id'] == x['id']],
                                               user_request['user_service_choice'],
                                               Service_types), result_type='expand', axis=1)

        score_blocks = pd.DataFrame({
            'scaled_population': Base_Layer_Blocks['population'].apply(
                lambda x: scaler(x, Base_Layer_Blocks['population'].min(), Base_Layer_Blocks['population'].max(),
                                 {'min': 1, 'max': 100})),
            'scaled_mean_unit_provision': Base_Layer_Blocks['mean_unit_provision'].apply(
                lambda x: invert_scaler(x, Base_Layer_Blocks['mean_unit_provision'].min(),
                                        Base_Layer_Blocks['mean_unit_provision'].max(),
                                        {'min': 1, 'max': 100})),
            'scaled_total_unit_resource': Base_Layer_Blocks['total_unit_resource'].apply(
                lambda x: invert_scaler(x, Base_Layer_Blocks['total_unit_resource'].min(),
                                        Base_Layer_Blocks['total_unit_resource'].max(), {'min': 1, 'max': 100}))})

        score_array = score_blocks[['scaled_population', 'scaled_mean_unit_provision',
                                    'scaled_total_unit_resource']].sum(axis=1)
        score_array_min, score_array_max = score_array.min(), score_array.max()
        Base_Layer_Blocks['score'] = score_array.apply(
            lambda x: scaler(x, score_array_min, score_array_max, {'min': 1, 'max': 100}))

        # Municipalities
        Base_Layer_Municipalities[['mean_unit_provision',
                                   'total_unit_resource']] = Base_Layer_Municipalities.apply(
            lambda x: service_score_properties(choice_subset[choice_subset['mo_id'] == x['id']],
                                               user_request['user_service_choice'], Service_types),
            result_type='expand', axis=1)
        score_municipalities = pd.DataFrame({
            'scaled_population': Base_Layer_Municipalities['population'].apply(
                lambda x: scaler(x, Base_Layer_Municipalities['population'].min(),
                                 Base_Layer_Municipalities['population'].max(), {'min': 1, 'max': 100})),
            'scaled_mean_unit_provision': Base_Layer_Municipalities['mean_unit_provision'].apply(
                lambda x: invert_scaler(x, Base_Layer_Municipalities['mean_unit_provision'].min(),
                                        Base_Layer_Municipalities['mean_unit_provision'].max(), {'min': 1, 'max': 100})),
            'scaled_total_unit_resource': Base_Layer_Municipalities['total_unit_resource'].apply(
                lambda x: invert_scaler(x, Base_Layer_Municipalities['total_unit_resource'].min(),
                                        Base_Layer_Municipalities['total_unit_resource'].max(), {'min': 1, 'max': 100}))})

        score_array = score_municipalities[['scaled_population', 'scaled_mean_unit_provision',
                                            'scaled_total_unit_resource']].sum(axis=1)
        score_array_min, score_array_max = score_array.min(), score_array.max()
        Base_Layer_Municipalities['score'] = score_array.apply(lambda x: scaler(x, score_array_min, score_array_max,
                                                                                {'min': 1, 'max': 100}))

        user_ads_rent_selection = Commercial_rent_ads[
            Commercial_rent_ads['square_m2'].between(int(user_request['user_unit_square_min']),
                                                     int(user_request['user_unit_square_max']))]

        response_json = {'municipalities': eval(Base_Layer_Municipalities.fillna('0').to_crs(4326).to_json()),
                         'blocks': eval(Base_Layer_Blocks.fillna('0').to_crs(4326).to_json()),
                         'rent_ads': eval(user_ads_rent_selection.fillna('0').to_crs(4326).to_json())}

        return response_json

    # ######################################### Wellbeing ##############################################
    def get_wellbeing(self, BCAM, living_situation_id=None, user_service_types=None, area=None,
                      provision_type="calculated", city="Saint_Petersburg", wellbeing_option=None):
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
        provision_columns = houses.filter(regex="provision")

        available_service_type = [t.split("_provision")[0] for t in provision_columns.columns]
        service_coef = service_coef[service_coef["service_code"].isin(available_service_type)]
        houses["wellbeing"] = provision_columns.apply(lambda x: self.calculate_wellbeing(x, service_coef), axis=1)
        houses["mean_provision"] = provision_columns.apply(lambda x: x[x != "None"].mean() if len(x[x != "None"]) > 0 else 0, axis=1)
        houses = houses.drop(list(provision_columns.columns) + list(houses.filter(regex="demand").columns) +
                             list(houses.filter(regex="num_available_services").columns), axis=1)

        if wellbeing_option:
            houses = houses[houses["wellbeing"].between(*wellbeing_option)]
            # PLUG!!! There must be slice by functional object id for services

        return {"houses": eval(houses.reset_index(drop=True).fillna("None").to_crs(4326).to_json()),
                "services": eval(services.reset_index(drop=True).fillna("None").to_crs(4326).to_json())}

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

        provision_columns = houses.filter(regex="provision")
        set_demand_columns = list(houses.filter(regex="demand").columns)
        set_num_service_columns = list(houses.filter(regex="num_available_services").columns)

        available_service_type = [t.split("_provision")[0] for t in provision_columns.columns]
        service_coef = service_coef[service_coef["service_code"].isin(available_service_type)]
        houses["wellbeing"] = provision_columns.apply(lambda x: self.calculate_wellbeing(x, service_coef), axis=1)

        if object_type == "house":
            houses["mean_provision"] = provision_columns.apply(lambda x: x[x != "None"].mean() if len(x[x != "None"]) > 0 else 0, axis=1)
            houses = houses.drop(set_demand_columns + set_num_service_columns + list(provision_columns.columns), axis=1)
            service_types_info = self.calculate_wellbeing(provision_columns.iloc[0], service_coef, get_provision=True)

        elif object_type == "service":
            service_type = services.iloc[0]["city_service_type"]
            service_code = city_inf_model.get_service_code(service_type)
            drop_col = [col for col in set_demand_columns if service_code not in col] + \
                       [col for col in set_num_service_columns if service_code not in col] + \
                       [col for col in provision_columns.columns if service_code not in col]
            houses = houses.drop(drop_col, axis=1)
            isochrone = gpd.GeoDataFrame.from_features(objects["isochrone"]).set_crs(4326)

        outcome_dict = {"houses": eval(houses.reset_index(drop=True).fillna("None").to_crs(4326).to_json()),
                        "services": eval(services.reset_index(drop=True).fillna("None").to_crs(4326).to_json())}

        if "service_types_info" in locals():
            outcome_dict["service_types"] = eval(service_types_info.to_json())
        elif "isochrone" in locals():
            outcome_dict["isochrone"] = eval(isochrone.to_json())
        return outcome_dict

    def calculate_wellbeing(self, loc, coef_df, get_provision=False):
        """
        :param loc: Series object containing provision evaluation with service type as index --> Series
        :param coef_df: DataFrame object containing columns with service types and coefficients --> DataFrame
        :param get_provision: option that define a return --> bool (default False)
                False - wellbeing evaluation for house as int,
                True - DataFrame with columns 'provision', 'coefficient' and 'wellbeing' for service types
        :return: see above
        """

        provision = loc.sort_index()
        provision.index = [idx.split("_provision_")[0] for idx in provision.index]
        available_type = provision != 'None'
        provision = provision[available_type]
        coef_df = coef_df.sort_values(by="service_code").set_index("service_code")["evaluation"][available_type]

        coef = list(coef_df)
        provision = list(provision)
        weighted_provision = [1 + 2 * coef[i] * (-1 + provision[i]) if coef[i] <= 0.5
                              else provision[i] ** (8 * coef[i] - 3) for i in range(len(provision))]
        if get_provision:
            return pd.DataFrame({"service_code": list(coef_df.index),
                                 "provision": provision, "coefficient": coef, "wellbeing": weighted_provision})
        else:
            return min(weighted_provision) if len(weighted_provision) > 0 else 0

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








