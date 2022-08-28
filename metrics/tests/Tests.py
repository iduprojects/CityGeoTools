import requests
import pandas as pd
import numpy as np
import warnings
import os
import geopandas as gpd
import json
import random
import shapely

from tqdm import tqdm

# TODO 1: Tests for house_selection and service_location
# TODO 2: Using postgres api or sql queries to get data to automaticaly create user request.
# TODO 3: Simplify metric tests. Create unified functions, e.g. for generating areas.

path = os.getcwd().split("Tests")[0]
services_spb = gpd.read_file(path + "Data/Services/services.geojson")
buildings_spb = gpd.read_file(path+ "Data/Buildings/houses_provision.geojson")
lst_serv_spb = services_spb["service_code"].unique()

mongo_address = "http://10.32.1.65:5100"
services_kr = requests.get(mongo_address + "/uploads/infrastructure/services").json()
services_kr = gpd.GeoDataFrame.from_features(services_kr)
lst_serv_kr = services_kr["service_code"].unique()

lst_serv = list(set(lst_serv_kr) & set(lst_serv_spb))

mo = requests.get('http://10.32.1.62:1244/api/city/1/municipalities').json()
blocks = requests.get('http://10.32.1.62:1244/api/city/1/blocks').json()
service_types_all = requests.get('http://10.32.1.62:1244/api/list/city_service_types/').json()
districs = requests.get(f'http://10.32.1.62:1244/api/city/1/administrative_units').json()

diversity = gpd.read_file(path + "/Data/Blocks/Blocks_Diversity.geojson")
diversity_services = diversity.filter(regex="_diversity").columns
diversity_service = [s_type.split("_diversity")[0] for s_type in diversity_services]


class TestError(Exception):

    def __init__(self, endpoint, status_code, request):
        self.endpoint = endpoint
        self.request = request
        self.status_code = status_code
        self.message = f"Endpoint test ({endpoint}) failed with status code {self.status_code}.\n{self.request}"
        super().__init__(self.message)


def check_metric(func, ip, n, method=None, *args):
    """
    Test interface
    :param func: test function  -> function
    :param ip: ip address where container is running -> str
    :param n: number of tests -> int
    :param method: endpoint -> str
    """
    endpoint = func.__name__.split("_test")[0] + "/" + method if method else func.__name__.split("_test")[0]
    for i in tqdm(range(1, n + 1), desc=endpoint):

        request, response = func(ip, method, *args) if method else func(ip, *args)
        status_code = response.status_code
        # if status_code == 400:
        #     warnings.warn("\nError 404 from:" + json.dumps(request))
        if status_code != 200 and status_code != 400:
            raise TestError(endpoint, status_code, request)


def get_random_area(mo, district):
    pass


def connectivity_viewer_test(ip, mo):

    city = ["Saint_Petersburg"]

    mo_choice = np.random.choice(mo)
    coords = mo_choice["geometry"]["coordinates"]

    # Endpoint: connectivity_calculations/connectivity_viewer
    request = {
        "type": "FeatureCollection",
        "name": "test_area",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {"type": "Feature", "properties": {"id": 1},
             "geometry": {"type": "Polygon",
                          "coordinates": coords[0]}}]}

    response = requests.post(f'http://{ip}/connectivity_calculations/connectivity_viewer', data=json.dumps(request))

    return request, response


def pedastrian_walk_traffics_test(ip, blocks):

    city = ["Saint_Petersburg"]

    blocks_choice = np.random.choice(blocks)
    coords = blocks_choice["geometry"]["coordinates"]

    # Endpoint: connectivity_calculations/connectivity_viewer
    request = {'crs': {'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
                       'type': 'name'},
               'features': [{'geometry': {'coordinates': coords[0],
                                          'type': 'Polygon'},
                             'properties': {},
                             'type': 'Feature'}]}

    response = requests.post(f'http://{ip}/pedastrian_walk_traffics/pedastrian_walk_traffics_calculation',
                             data=json.dumps(request))

    return request, response


def mobility_analysis_test(ip, method=None):
    travel_type = ["walk", "drive"]
    city = ["Saint_Petersburg", "Krasnodar"]
    travel_type_choice = np.random.choice(travel_type)
    city_choice = np.random.choice(city)

    if city_choice == "Saint_Petersburg":
        x_range, y_range = np.arange(59.82, 60.07, 0.01), np.arange(30.17, 30.55, 0.01)

    elif city_choice == "Krasnodar":
        x_range, y_range = np.arange(45, 45.15, 0.01), np.arange(38.89, 39.12, 0.01)

    x_from, x_to = np.random.choice(x_range), np.random.choice(x_range)
    y_from, y_to = np.random.choice(y_range), np.random.choice(y_range)

    unit = ["times", "meters"]
    unit_choice = np.random.choice(unit)
    range_time, range_meters = range(5, 100, 5), range(1000, 5000, 500)
    travel_time = np.random.choice(range_time) if unit_choice == "times" else np.random.choice(range_meters)

    # Endpoint: mobility_analysis/routes
    if method == "routes":
        request = f'http://{ip}/mobility_analysis/routes?' + \
                  f'city={city_choice}&x_from={x_from}&y_from={y_from}&x_to={x_to}&y_to={y_to}&' + \
                  f'travel_type={travel_type_choice}'

        response = requests.get(request)

    # Endpoint: mobility_analysis/isochrones
    elif method == "isochrones":
        request = f'http://{ip}/mobility_analysis/isochrones?' + \
                  f'city={city_choice}&x_from={x_from}&y_from={y_from}&' + \
                  f'{unit_choice}={[travel_time]}&travel_type={travel_type_choice}'

        response = requests.get(request)

    # Endpoint: mobility_analysis/isochrones
    elif method == "multimodal_isochrones":
        request = f'http://{ip}/mobility_analysis/isochrones?' + \
                  f'city={city_choice}&x_from={x_from}&y_from={y_from}&' + \
                  f'travel_time={travel_time}&travel_type=public_transport'
        response = requests.get(request)

    return request, response


def visibility_analysis_test(ip):
    city = ["Saint_Petersburg", "Krasnodar"]
    city_choice = np.random.choice(city)
    view_distance = np.arange(50, 2000, 50)
    view_distance_choice = np.random.choice(view_distance)

    if city_choice == "Saint_Petersburg":
        x_range, y_range = np.arange(59.82, 60.07, 0.01), np.arange(30.17, 30.55, 0.01)
    elif city_choice == "Krasnodar":
        x_range, y_range = np.arange(45, 45.15, 0.01), np.arange(38.89, 39.12, 0.01)
    x_from, y_from = np.random.choice(x_range), np.random.choice(y_range)

    # Endpoint: visibility_analysis/visibility_analysis
    request = f'http://{ip}/Visibility_analysis/Visibility_analysis?city={city_choice}&' + \
              f'x_from={x_from}&y_from={y_from}&view_distance={view_distance_choice}'
    response = requests.get(request)

    return request, response


def voronoi_test(ip):
    city = ["Saint_Petersburg", "Krasnodar"]
    city_choice = np.random.choice(city)
    n_points = np.random.choice(range(3, 10))

    if city_choice == "Saint_Petersburg":
        x_range, y_range = np.arange(3360838, 3393559, 100), np.arange(8363855, 8420072, 100)
        crs = 3857
    elif city_choice == "Krasnodar":
        x_range, y_range = np.arange(493019, 513075, 100), np.arange(4990561, 4997849, 100)
        crs = 32637

    features = []
    for i in range(n_points):
        dict_point = {"geometry": {"coordinates": [float(np.random.choice(x_range)),
                                                   float(np.random.choice(y_range))], "type": "Point"},
                      "properties": {"weight": float(np.random.choice(np.arange(1, 10, 0.1)))},
                      "type": 'Feature'}
        features.append(dict_point)

    request = {
        "city": city_choice,
        "geojson": {
            'crs': {'properties': {'name': 'urn:ogc:def:crs:EPSG::' + str(crs)}, 'type': 'name'},
            'features': features,
            'name': 'test',
            'type': 'FeatureCollection'
        }
    }

    # Endpoint: voronoi/Weighted_voronoi_calculation
    response = requests.post(f'http://{ip}/voronoi/Weighted_voronoi_calculation', data=json.dumps(request))

    return request, response


def get_services_test(ip, service_types_all, blocks):
    city = ["Saint_Petersburg", "Krasnodar"]
    city_choice = np.random.choice(city)
    city_ids = {"Saint_Petersburg": 1, "Krasnodar": 2}
    city_id = city_ids[city_choice]

    service_types = np.random.choice(service_types_all, random.randint(1, len(lst_serv)), replace=False)

    area_types = {"mo": "municipalities", "district": "administrative_units", "block": "blocks"}
    area_type = np.random.choice(list(area_types.keys()))

    if area_type == "block" and city_choice == "Krasnodar":
        area = blocks
        area_ids = list(gpd.GeoDataFrame(area)["id"])
    else:
        area = requests.get(f'http://10.32.1.62:1244/api/city/{city_id}/{area_types[area_type]}').json()
        area_ids = list(pd.DataFrame(area)["id"])
    area_id = np.random.choice(area_ids)

    # Endpoint: /services_clusterization/get_services and /blocks_clusterization/get_services
    request = {"city": city_choice,
               "param": {"service_types": list(service_types),
                         "area": {area_type: float(area_id)}}}

    response = requests.post(f"http://{ip}/services_clusterization/get_services", data=json.dumps(request))

    return request, response


def services_clusterization_test(ip, service_types_all):
    city = ["Saint_Petersburg", "Krasnodar"]
    city_choice = np.random.choice(city)
    city_ids = {"Saint_Petersburg": 1,
                "Krasnodar": 2}
    city_id = city_ids[city_choice]

    service_types = np.random.choice(service_types_all, random.randint(1, len(service_types_all)), replace=False)

    criterion = ["default", "maxclust"]
    condition = np.random.choice(criterion)

    value = random.randint(100, 10000) if condition == "default" else random.randint(1, 10000)
    n_std = random.randint(1, 10)

    area_types = {"mo": "municipalities", "district": "administrative_units"}
    area_type = np.random.choice(list(area_types.keys()))

    area = requests.get(f'http://10.32.1.62:1244/api/city/{city_id}/{area_types[area_type]}').json()
    area_ids = list(pd.DataFrame(area)["id"])
    area_id = np.random.choice(area_ids)

    # Endpoint: services_clusterization/get_clusters_polygons
    request = {"city": city_choice,
               "param": {"service_types": list(service_types),
                         "condition": {condition: value},
                         "n_std": n_std,
                         "area": {area_type: float(area_id)}}}

    response = requests.post(f"http://{ip}/services_clusterization/get_clusters_polygons", data=json.dumps(request))

    return request, response


def blocks_clusterization_test(ip, method, service_types_all):
    city = ["Saint_Petersburg", "Krasnodar"]
    city_choice = np.random.choice(city)

    service_types = np.random.choice(service_types_all, random.randint(1, 10), replace=False)

    cluster_type = ["default", "user_specified"]
    cluster_type_choice = np.random.choice(cluster_type)

    # Endpoint: blocks_clusterization/get_blocks and blocks_clusterization/get_dendrogram
    if cluster_type_choice == "default":
        request = {"city": city_choice, "param": {"service_types": list(service_types)}}
    else:
        request = {"city": city_choice,
                   "param": {"service_types": list(service_types),
                             "clusters_number": random.randint(1, 100)}}

    response = requests.post(f'http://{ip}/blocks_clusterization/{method}', data=json.dumps(request))

    return request, response


def house_location_test(ip, method, services, buildings, service_types_all):
    city = ["Saint_Petersburg"]
    city_choice = np.random.choice(city)

    service_types = np.random.choice(service_types_all, random.randint(1, len(service_types_all)), replace=False)
    area = random.randint(1, 100000)
    population = random.randint(1, 10000)
    floors = random.randint(1, 50)

    # Endpoint: /house_location/house_location
    if method == "house_location":
        request = {"area": area,
                   "floors": floors,
                   "population": population,
                   "service_types": list(service_types)}

    # Endpoint: /house_location/block_provision
    elif method == "block_provision":
        service_intype = services[services["service_code"].isin(service_types)]
        available_id = service_intype[service_intype["block_id"].isin(buildings["block_id"])]["block_id"].dropna().unique()

        block_id = np.random.choice(list(available_id))
        request = {"block_id": int(block_id),
                   "population": population,
                   "service_types": list(service_types)}

    response = requests.post(f'http://{ip}/house_location/{method}', data=json.dumps(request))

    return request, response


def house_selection_test(ip):

    # Endpoint: house_selection/House_selection
    request = {"user_selected_significant_services": ["Урна", "Автозаправка"],
               "user_selected_unsignificant_services": ["Кафе/столовая"],
               "user_social_group_selection": "Дети младшего школьного возраста (7-11)",
               "user_price_preferences": [20000, 30000]}
    response = requests.post(f'http://{ip}/house_selection/House_selection', data=json.dumps(request))

    return request, response


def service_location_test(ip):

    # Endpoint: /service_location/service_location
    request = {"user_service_choice": "Кафе/столовая",
               "user_unit_square_min": float(np.random.choice(range(10, 30))),
               "user_unit_square_max": float(np.random.choice(range(30, 100)))}
    response = requests.post(f'http://{ip}/service_location/service_location', data=json.dumps(request))

    return request, response


def spacematrix_test(ip, method, mo, districts):
    city = ["Saint_Petersburg"]
    city_choice = np.random.choice(city)
    border_type = ["polygon", "municipalities", "districts"]
    border_type_choice = np.random.choice(border_type)
    indexes = ["FSI", "GSI", "L", "OSR", "MXI", "Spacematrix"]
    index_choice = np.random.choice(indexes)

    if border_type_choice == "polygon":
        value_x = np.arange(30, 30.5, 0.001)
        value_y = np.arange(59.6, 60.1, 0.001)
        box = shapely.geometry.box(np.random.choice(value_x), np.random.choice(value_y),
                                   np.random.choice(value_x), np.random.choice(value_y))
        value = list(box.exterior.coords)

    elif border_type_choice == "municipalities":
        area = gpd.GeoDataFrame(mo)
        name_mo = area["name"].unique()
        value = np.random.choice(name_mo)

    elif border_type_choice == "districts":
        area = gpd.GeoDataFrame(districts)
        name_districts = area["name"].unique()
        value = np.random.choice(name_districts)

    # Endpoint: /spacematrix/get_objects
    if method == "get_objects":
        user_request = {border_type_choice: value}
        response = requests.get(f'http://{ip}/spacematrix/{method}?{border_type_choice}={value}')

    # Endpoint: /spacematrix/get_indices
    elif method == "get_indices":
        user_request = {border_type_choice: value,
                        "index": index_choice}
        response = requests.get(f'http://{ip}/spacematrix/{method}?{border_type_choice}={value}&index={index_choice}')

    return user_request, response


def get_diversity_test(ip, service_types_all):
    city = ["Saint_Petersburg"]
    city_choice = np.random.choice(city)
    service_type = np.random.choice(service_types_all)

    # Endpoint: /diversity/diversity
    request = {"service_type": service_type}

    response = requests.get(f'http://{ip}/diversity/diversity?service_type={service_type}')

    return request, response


def provision_test(ip, method, service_types_all, mo, districts):

    city = "Saint_Petersburg"

    if method == "get_provision":
        service_type = np.random.choice(service_types_all)

        border_type = ["polygon", "mo", "district"]
        border_type_choice = np.random.choice(border_type)
        if border_type_choice == "polygon":
            mo_choice = np.random.choice(mo)
            coords = mo_choice["geometry"]["coordinates"]
            value = {
                "type": "FeatureCollection",
                "name": "test_area",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": [
                    {"type": "Feature", "properties": {"id": 1},
                     "geometry": {"type": "Polygon",
                                  "coordinates": coords[0]}}]}

        elif border_type_choice == "mo":
            area = gpd.GeoDataFrame(mo)
            name_mo = area["name"].unique()
            value = np.random.choice(name_mo)

        elif border_type_choice == "district":
            area = gpd.GeoDataFrame(districts)
            name_districts = area["name"].unique()
            value = np.random.choice(name_districts)

        request = {"city": city,
                   "service_type": service_type,
                   "area": {border_type_choice: value}}

        response = requests.post(f'http://{ip}/provision/get_provision', data=json.dumps(request))

        return request, response


if __name__ == '__main__':
    # ip = '10.32.1.65:5000'
    ip = '127.0.0.1:5000'
    n = 10

    # check_metric(connectivity_viewer_test, ip, n, None, mo)
    # check_metric(pedastrian_walk_traffics_test, ip, n, None, blocks)
    #
    check_metric(mobility_analysis_test, ip=ip, method='routes', n=n)
    # check_metric(mobility_analysis_test, ip=ip, method='isochrones', n=n)
    # check_metric(mobility_analysis_test, ip=ip, method='multimodal_isochrones', n=n)
    #
    # check_metric(visibility_analysis_test, ip, n)
    # check_metric(voronoi_test, ip, n)
    #
    # check_metric(get_services_test, ip, n, None, lst_serv, blocks)
    # check_metric(services_clusterization_test, ip, n, None, lst_serv)
    # check_metric(blocks_clusterization_test, ip, n, 'get_blocks', lst_serv)
    # check_metric(blocks_clusterization_test, ip, n, 'get_dendrogram', lst_serv)
    #
    # check_metric(get_diversity_test, ip, n, None, diversity_service)
    #
    # check_metric(house_location_test, ip, n, 'house_location', services_spb, buildings_spb, lst_serv)
    # check_metric(house_location_test, ip, n, 'block_provision', services_spb, buildings_spb, lst_serv)
    #
    # check_metric(house_selection_test, ip, 2)
    # check_metric(service_location_test, ip, 2)
    #
    # check_metric(spacematrix_test, ip, n, 'get_objects', mo, districs)
    # check_metric(spacematrix_test, ip, n, 'get_indices', mo, districs)
    #
    # check_metric(provision_test, ip, n, 'get_provision', lst_serv, mo, districs)
