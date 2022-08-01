import os
from enum import auto
from typing import List

import geopandas as gpd
from fastapi import APIRouter, HTTPException, status, Body, Depends
from fastapi.responses import StreamingResponse
from geojson_pydantic import FeatureCollection

from app import enums, schemas
from app.core.config import settings
from Calculations import utils
from Calculations.City_Metrics_Methods import *
from Calculations.Basics.Basics_City_Analysis_Methods import Basics_City_Analysis_Methods
from Data.cities_dictionary import cities_model, cities_crs

router = APIRouter()

CMM = City_Metrics_Methods(cities_model, cities_crs)
BCAM = Basics_City_Analysis_Methods(cities_model, cities_crs)


class Tags(str, enums.AutoName):
    def _generate_next_value_(name, start, count, last_values):
        return name

    trafics_calculation = auto()
    mobility_analysis = auto()
    visibility_analysis = auto()
    weighted_voronoi = auto()
    blocks_clusterization = auto()
    services_clusterization = auto()
    spacematrix = auto()
    diversity = auto()
    provision = auto()
    well_being = auto()


@router.get("/")
async def read_root():
    return {"Hello": "World"}

@router.post('/pedastrian_walk_traffics/pedastrian_walk_traffics_calculation', 
            response_model=schemas.PedastrianWalkTrafficsCalculationOut, tags=[Tags.trafics_calculation])
def pedastrian_walk_traffics_calculation(query_params: schemas.PedastrianWalkTrafficsCalculationIn):
    city_model = cities_model[query_params.city]
    result = TraficCalculator(city_model).get_trafic_calculation(query_params.geojson.dict())
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No living houses in the specified area"
        )
    return result


@router.get("/mobility_analysis/routes", response_model=schemas.MobilityAnalysisRoutesOut,
            tags=[Tags.mobility_analysis])
async def mobility_analysis_routes(query_params: schemas.MobilityAnalysisRoutesQueryParams = Depends()):
    result = BCAM.Route_Between_Two_Points(
        city=query_params.city, travel_type=query_params.travel_type,
        x_from=query_params.x_from, y_from=query_params.y_from,
        x_to=query_params.x_to, y_to=query_params.y_to,
        reproject=query_params.reproject
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Path between given points absents",
        )
    return result


@router.get("/mobility_analysis/isochrones", response_model=schemas.MobilityAnalysisIsochronesOut,
            tags=[Tags.mobility_analysis])
async def mobility_analysis_isochrones(query_params: schemas.MobilityAnalysisIsochronesQueryParams = Depends()):
    if (query_params.travel_type == enums.MobilityAnalysisIsochronesTravelTypeEnum.PUBLIC_TRANSPORT) and \
            (query_params.weight_type == enums.MobilityAnalysisIsochronesWeightTypeEnum.METER):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The weight type isn't supported for public transport isochrones."
        )

    # todo решить вопрос универсальной подстановкой методов
    if query_params.travel_type == enums.MobilityAnalysisIsochronesTravelTypeEnum.PUBLIC_TRANSPORT:
        result = BCAM.transport_isochrone(
            city=query_params.city, travel_type=query_params.travel_type,
            weight_type=query_params.weight_type, weight_value=query_params.weight_value,
            x_from=query_params.x_from, y_from=query_params.y_from,
        )
    else:
        result = BCAM.walk_drive_isochrone(
            city=query_params.city, travel_type=query_params.travel_type,
            weight_type=query_params.weight_type, weight_value=query_params.weight_value,
            x_from=query_params.x_from, y_from=query_params.y_from,
        )

    return result


@router.get("/Visibility_analysis/Visibility_analysis", response_model=FeatureCollection,
            tags=[Tags.visibility_analysis])
async def Visibility_analisys(query_params: schemas.VisibilityAnalisysQueryParams = Depends()):
    request_points = [[query_params.x_from, query_params.y_from]]
    to_crs = cities_model[query_params.city].city_crs
    request_point = utils.request_points_project(request_points, 4326, to_crs)[0]
    return CMM.Visibility_Analysis(query_params.city, request_point, query_params.view_distance)


@router.post("/voronoi/Weighted_voronoi_calculation", response_model=schemas.WeightedVoronoiCalculationOut,
             tags=[Tags.weighted_voronoi])
async def Weighted_voronoi_calculation(user_request: schemas.WeightedVoronoiCalculationIn):
    """
    In user request:
    :param: city -> srt
    :param: geojson (containing points with weight) -> geojson
    :return: voronoi polygons around points and polygons of deficit_zones ->geojson
    """
    return CMM.Weighted_Voronoi(user_request.dict())


@router.post("/blocks_clusterization/get_blocks", response_model=FeatureCollection,
             tags=[Tags.blocks_clusterization])
async def get_blocks_calculations(user_request: schemas.BlocksClusterizationGetBlocks):
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: clusters_number, service_types) -> json
    :return: block polygons with cluster labels and cluster parameters -> geojson
    """
    result = BCAM.Blocks_Clusterization(user_request.city, user_request.param.dict(), method="get_blocks")
    return FeatureCollection.parse_raw(result)  # todo return json dict not str


@router.post("/blocks_clusterization/get_services", response_model=FeatureCollection,
             tags=[Tags.blocks_clusterization])
async def get_services_calculations(user_request: schemas.BlocksClusterizationGetServices):
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: service_types, area) -> json
    :return: service points in specified blocks -> geojson
    """
    result = BCAM.Get_Services(user_request.city, user_request.param.dict())
    return FeatureCollection.parse_raw(result)  # todo return json dict not str


@router.post("/blocks_clusterization/get_dendrogram",
             responses={
              200: {
                  "content": {"image/png": {}}
              }
          },
             response_class=StreamingResponse,
             tags=[Tags.blocks_clusterization])
async def get_dendrogram(user_request: schemas.BlocksClusterizationGetBlocks):
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: clusters_number, service_types) -> json
    :return: dendrogram image -> byte str
    """
    result = BCAM.Blocks_Clusterization(user_request.city, user_request.param.dict(), method="get_dendrogram")
    return StreamingResponse(content=result, media_type="image/png")


@router.post("/services_clusterization/get_services", response_model=FeatureCollection,
             tags=[Tags.services_clusterization])
async def get_services(user_request: schemas.ServicesClusterizationGetServices):
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: service_types, area) -> json
    :return: service points in specified blocks -> geojson
    """
    result = BCAM.Get_Services(user_request.city, user_request.param.dict())
    return FeatureCollection.parse_raw(result)  # todo return json dict not str


@router.post("/services_clusterization/get_clusters_polygons", response_model=FeatureCollection,
             tags=[Tags.services_clusterization])
async def get_services_clusterization(user_request: schemas.ServicesClusterizationGetClustersPolygonsIn):
    """
    In user request:
    :param: city -> srt
    :param: user json (with keys: service_types, area, condition, n_std) -> json
    :return: polygons of point cluster -> geojson
    """
    result = BCAM.Services_Clusterization(user_request.city, user_request.param.dict())

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enough objects to cluster"
        )

    return FeatureCollection.parse_raw(result)  # todo return json dict not str


# todo Spacematrix


@router.get("/diversity/diversity", response_model=schemas.DiversityDiversityOut,
            tags=[Tags.diversity])
async def get_diversity(query_params: schemas.DiversityDiversityQueryParams = Depends()):  # todo validate service_type?
    """
    In user request:
    :param: service_type -> str
    :return: polygons of blocks/municipalities -> geojson
    """
    result = BCAM.Get_Diversity(query_params.service_type)
    return result


@router.post("/provision/get_provision", response_model=schemas.ProvisionGetProvisionOut,
             tags=[Tags.provision])
async def get_provision(user_request: schemas.ProvisionGetProvisionIn):
    """
    In user request:
    required params: service_type, area, provision_type
    optional params: city, without_services_options, load_options, provision_option
    :return: dict of FeatureCollections houses and services
    """
    result = BCAM.get_provision(**user_request.dict())
    return result


@router.post("/provision/get_info", response_model=schemas.ProvisionGetInfoOut,
             tags=[Tags.provision])
async def get_provision_info(user_request: schemas.ProvisionGetInfoIn):
    """
    In user request:
    required params: object_type, functional_object_id, service_type, provision_type
    :return: dict of FeatureCollections of houses, services and isochrone (not for all request)
    """
    result = BCAM.get_provision_info(**user_request.dict())
    return result


@router.post("/wellbeing/get_wellbeing", response_model=schemas.WellbeingGetWellbeingOut,
             tags=[Tags.well_being])
async def get_wellbeing(user_request: schemas.WellbeingGetWellbeingIn):
    """
    In user request:
    required params: provision_type and either living_situation_id or user_service_types
    :return: dict of FeatureCollections houses and services
    """
    result = CMM.get_wellbeing(BCAM, **user_request.dict())
    return result


@router.post("/wellbeing/get_wellbeing_info", response_model=schemas.WellbeingGetWellbeingInfoOut,
             tags=[Tags.well_being])
async def get_wellbeing_info(user_request: schemas.WellbeingGetWellbeingInfoIn):
    """
    In user request:
    required params: provision_type, object_type, functional_object_id and either living_situation_id or user_service_types
    :return: dict of FeatureCollections of houses, services, isochrone (not for all request) and service types as json (not for all request)
    """
    result = CMM.get_wellbeing_info(BCAM, **user_request.dict())
    return result

