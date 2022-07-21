import requests
import json
import os
import ast
import geopandas as gpd

from flask import Flask, jsonify, request, send_file, abort
from flask_cors import CORS, cross_origin
from Calculations.City_Metrics_Methods import City_Metrics_Methods
from Calculations.Basics.Basics_City_Analysis_Methods import Basics_City_Analysis_Methods
from Data.Cities_dictionary import cities_model, cities_crs, cities_metrics, cities_name


app = Flask(__name__)
path = os.getcwd()
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

CMM = City_Metrics_Methods(cities_model, cities_crs)
BCAM = Basics_City_Analysis_Methods(cities_model, cities_crs)

''' 
Mostly CMM class methods take as input only user request that parse inside each method.
BCAM class methods take as input parsed user request because they can be called from CMM class methods.
'''


# ############## Documentation ##############
@app.route('/documentation', methods=["GET"])
@cross_origin()
def docs():
    """
    :return: metrics documentation
    """
    with open(path + "/docs.json") as f:
        docs = json.load(f)
    result = docs[request.args.get('method')] if request.args.get('method') else list(docs.keys())
    return jsonify(result)


# ############## Available cities ##############
@app.route('/cities/<method>', methods=["GET"])
@cross_origin()
def cities(method):
    """
    In user request:
    :param: method (city) -> str or None
    :return: /cities/metrics - dict (or list) of available cities metrics
            /cities/crs - dict (or str) of cities crs
            /cities/name - list of available cities
    """
    if method == "metrics":
        result = cities_metrics[request.args.get('city')] if request.args.get('city') else cities_metrics
    elif method == "crs":
        result = cities_crs[request.args.get('city')] if request.args.get('city') else cities_crs
    elif method == "name":
        result = cities_name

    return jsonify(result)


# #######################  Connectivity calculations  ########################
@app.route('/connectivity_calculations/connectivity_viewer', methods=["POST"])
@cross_origin()
def connectivity_viewer():
    """
    In user request:
    :param: geojson (containing polygon) -> geojson
    :return: polygon and point -> geojson
    """
    result = CMM.Connectivity_Viewer(request.get_json(force=True))
    return jsonify(result)


# ############################  Trafics calculation  #############################
@app.route('/pedastrian_walk_traffics/pedastrian_walk_traffics_calculation', methods=["POST"])
@cross_origin()
def pedastrian_walk_traffics_calculation():
    """
    In user request:
    :param: geojson (containing polygon) -> geojson
    :return: linestrings -> geojson
    """
    result = CMM.Trafic_Area_Calculator(BCAM, request.get_json(force=True))
    if result:
        return jsonify(result)
    else:
        abort(400, description="No living houses in the specified area")


# ############################# Mobility analysis ################################
@app.route('/mobility_analysis/routes', methods=["GET"])
@cross_origin()
def mobility_analysis_routes():
    """
    In user request:
    required params: city, travel_type, x_from, y_from, x_to, y_to, reproject
    :return: geojson route linestring
    """
    result = BCAM.Route_Between_Two_Points(**request.args)
    if result is None:
        abort(422, "Path between given points absents")
    return jsonify(result)


@app.route('/mobility_analysis/isochrones', methods=["GET"])
@cross_origin()
def mobility_analysis_isochrones():
    """
    In user request:
    required params for transport isochrone: city, travel_type, travel_time, x_from, y_from, weight_type, weight_value
    required params for walk/drive isochrone: city, travel_type, x_from, y_from, weight_type, weight_value
    :return: geojson containing isochrones
    """
    if (request.args.get('travel_type') == 'public_transport') and (request.args.get('weight_type') == 'meter'):
        abort(400, description="The weight type isn't supported for public transport isochrones.")

    if request.args.get('travel_type') == 'public_transport':
        result = BCAM.transport_isochrone(**request.args)
    else:
        result = BCAM.walk_drive_isochrone(**request.args)
    return jsonify(result)


# ##############################  Visibility analysis  ##############################
@app.route('/Visibility_analysis/Visibility_analysis', methods=["GET"])
@cross_origin()
def Visibility_analisys():
    """
    In user request:
    :param: x_from, y_from (viewpoint) -> float,
    :param: view_distance -> int
    :return: polygon of visibility -> geojson
    """
    request_points = [[float(request.args.get('x_from')), float(request.args.get('y_from'))]]
    city, view_distance = request.args.get('city'), float(request.args.get('view_distance'))

    request_points = BCAM.Request_Points_Project(city, request_points)
    result = CMM.Visibility_Analysis(city, request_points[0], view_distance)
    return jsonify(result)


# ##############################  Weighted voronoi  ##################################
@app.route('/voronoi/Weighted_voronoi_calculation', methods=["POST"])
@cross_origin()
def Weighted_voronoi_calculation():
    """
    In user request:
    :param: city -> srt
    :param: geojson (containing points with weight) -> geojson
    :return: voronoi polygons around points and polygons of deficit_zones ->geojson
    """
    result = CMM.Weighted_Voronoi(request.get_json(force=True))
    return jsonify(result)


# ############################## Instagram ##############################
@app.route('/instagram_concentration/get_squares', methods=["GET"])
@cross_origin()
def instagram_get_squares():
    # args = year - season - day_time
    year = str(request.args.get('year'))
    season = str(request.args.get('season'))
    day_time = str(request.args.get('day_time'))
    result = CMM.get_instagram_data(year, season, day_time)
    return jsonify(result)


# ############################  House location  ##############################
@app.route('/house_location/house_location', methods=["POST"])
@cross_origin()
def House_location_calculations():
    """
    In user request:
    :param: user json (with keys: area, floors, population, service_types) -> json
    :return: polygons of scored blocks/municipalities -> geojson
    """
    result = CMM.House_Location(request.get_json(force=True))
    return jsonify(result)


@app.route('/house_location/block_provision', methods=["POST"])
@cross_origin()
def block_provision_calculations():
    """
    In user request:
    :param: user json (with keys: block_id, population, service_types) -> json
    :return: block polygons (containing information about provision before/after) -> geojson
    """
    result = CMM.Block_Provision(request.get_json(force=True))
    if result:
        return jsonify(result)
    else:
        abort(400, description="There are no services in the blocks")


# ############################## House selection ##############################
@app.route('/house_selection/social_groups_list', methods=["GET"])
@cross_origin()
def get_social_groups_list():
    """
    :return: social groups names -> list
    """
    city_crs = cities_model["Saint_Petersburg"]
    return jsonify(city_crs.Social_groups)


@app.route('/house_selection/House_selection', methods=["POST"])
@cross_origin()
def House_selection_calculations():
    """
    In user request:
    :param: user json (with keys: user_selected_significant_services, user_selected_unsignificant_services,
                    user_social_group_selection, user_price_preferences) -> json
    :return: polygons of scored blocks/municipalities -> geojson
    """
    result = CMM.House_selection(request.get_json(force=True))
    return jsonify(result)


# ############################## Blocks clusterization ##############################
@app.route('/blocks_clusterization/get_blocks', methods=["POST"])
@cross_origin()
def get_blocks_calculations():
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: clusters_number, service_types) -> json
    :return: block polygons with cluster labels and cluster parameters -> geojson
    """
    city, param = request.get_json(force=True)["city"], request.get_json(force=True)["param"]
    result = BCAM.Blocks_Clusterization(city, param, method="get_blocks")
    return jsonify(json.loads(result))


@app.route('/blocks_clusterization/get_services', methods=["POST"])
@cross_origin()
def get_services_calculations():
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: service_types, area) -> json
    :return: service points in specified blocks -> geojson
    """
    city, param = request.get_json(force=True)["city"], request.get_json(force=True)["param"]
    result = BCAM.Get_Services(city, param)
    return jsonify(json.loads(result))


@app.route('/blocks_clusterization/get_dendrogram', methods=["POST"])
@cross_origin()
def get_dendrogram():
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: clusters_number, service_types) -> json
    :return: dendrogram image -> byte str
    """
    city, param = request.get_json(force=True)["city"], request.get_json(force=True)["param"]
    result = BCAM.Blocks_Clusterization(city, param, method="get_dendrogram")
    return send_file(result, mimetype="image/png")


# ############################## Services clusterization ##############################
@app.route('/services_clusterization/get_services', methods=["POST"])
@cross_origin()
def get_services():
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: service_types, area) -> json
    :return: service points in specified blocks -> geojson
    """
    city, param = request.get_json(force=True)["city"], request.get_json(force=True)["param"]
    result = BCAM.Get_Services(city, param)
    return jsonify(json.loads(result))


@app.route('/services_clusterization/get_clusters_polygons', methods=["POST"])
@cross_origin()
def get_services_clusterization():
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: service_types, area, condition, n_std) -> json
    :return: polygons of point cluster -> geojson
    """
    city, param = request.get_json(force=True)["city"], request.get_json(force=True)["param"]
    result = BCAM.Services_Clusterization(city, param)

    if result:
        return jsonify(json.loads(result))
    else:
        abort(400, description="Not enough objects to cluster")


# ############################## Services location ##############################
@app.route('/service_location/service_location', methods=["POST"])
@cross_origin()
def service_location():
    """
    In user request:
    :param: user json (with keys: service_types, area, condition, n_std) -> json
    :return: polygons of scored blocks/municipalities -> geojson
    """
    result = CMM.Service_location(request.get_json(force=True))
    return jsonify(result)


# ############################## Spacematrix ##############################
@app.route('/spacematrix/get_objects', methods=["GET"])
@cross_origin()
def spacematrix_objects():
    """
    In user request:
    :param: district/municipality (display area of metric results) -> str
            or polygon -> list of tuple of float pair numbers
    :return: block polygons and building polygons -> geojson
    """
    result = CMM.Get_Objects(request.args)
    return jsonify(json.loads(result[0]), json.loads(result[1]))

@app.route('/spacematrix/get_indices', methods=["GET"])
@cross_origin()
def spacematrix_indices():
    """
    In user request:
    :param: district/municipality (display area of metric results) -> str, or polygon -> list of tuple of float pair numbers
    :param: index (one of Spacematrix indices) -> str
    :return: block polygons (containing values of Spacematrix index) -> geojson
    """
    result = CMM.Get_Indices(request.args)
    return jsonify(json.loads(result))


# ############################# Diversity ##############################
@app.route('/diversity/diversity', methods=["GET"])
@cross_origin()
def get_diversity():
    """
    In user request:
    :param: service_type -> str
    :return: polygons of blocks/municipalities -> geojson
    """
    service_type = request.args['service_type']
    result = BCAM.Get_Diversity(service_type)
    return jsonify(result)


# ############################# Provision ##############################
@app.route('/provision/get_provision', methods=["POST"])
@cross_origin()
def get_provision():
    """
    In user request:
    required params: service_type, area, provision_type
    optional params: is_weighted, service_coef
    :return: dict of FeatureCollections houses and services
    """
    user_request = request.get_json(force=True)
    result = BCAM.get_provision(**user_request)
    if type(result) is tuple:
        abort(400, description=result[1])
    else:
        return jsonify(result)


@app.route('/provision/get_info', methods=["POST"])
@cross_origin()
def get_provision_info():
    """
    In user request:
    required params: object_type, functional_object_id, service_type, provision_type
    :return: dict of FeatureCollections of houses, services and isochrone (not for all request)
    """
    user_request = request.get_json(force=True)
    result = BCAM.get_provision_info(**user_request)
    if type(result) is tuple:
        abort(400, description=result[1])
    else:
        return jsonify(result)


@app.route('/provision/aggregate', methods=["POST"])
@cross_origin()
def get_provision_aggregated():
    """
    In user request:
    required params: service_types, area_type, provision_type
    optional params: is_weighted, service_coef
    :return: FeatureCollections of territorial units
    """
    user_request = request.get_json(force=True)
    result = BCAM.get_provision_aggregated(**user_request)
    return jsonify(result)

@app.route('/provision/top_objects', methods=["POST"])
@cross_origin()
def get_provision_top():
    """
    In user request:
    required params: service_types, area_type, area_value, provision_type
    optional params: is_weighted, service_coef
    :return: dict of FeatureCollections of houses, services
    """
    user_request = request.get_json(force=True)
    result = BCAM.get_top_provision_objects(**user_request)
    return jsonify(result)


# ######################### Well-being #############################
@app.route('/wellbeing/get_wellbeing', methods=["POST"])
@cross_origin()
def get_wellbeing():
    """
    In user request:
    required params: provision_type and either living_situation_id or user_service_types
    :return: dict of FeatureCollections houses and services
    """
    user_request = request.get_json(force=True)
    result = CMM.get_wellbeing(BCAM, **user_request)
    if type(result) is tuple:
        abort(400, description=result[1])
    else:
        return jsonify(result)


@app.route('/wellbeing/get_wellbeing_info', methods=["POST"])
@cross_origin()
def get_wellbeing_info():
    """
    In user request:
    required params: provision_type, object_type, functional_object_id and either living_situation_id or user_service_types
    :return: dict of FeatureCollections of houses, services, isochrone (not for all request) and service types as json (not for all request)
    """
    user_request = request.get_json(force=True)
    result = CMM.get_wellbeing_info(BCAM, **user_request)
    if type(result) is tuple:
        abort(400, description=result[1])
    else:
        return jsonify(result)


@app.route('/wellbeing/aggregate', methods=["POST"])
@cross_origin()
def get_wellbeing_aggregate():
    """
    In user request:
    required params: area_type, provision_type and either living_situation_id or user_service_types
    :return: FeatureCollections of houses
    """
    user_request = request.get_json(force=True)
    result = CMM.get_wellbeing_aggregated(BCAM, **user_request)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)

