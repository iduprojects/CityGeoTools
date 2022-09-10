from enum import auto

import geopandas as gpd
from fastapi import APIRouter, HTTPException, status, Body, Depends
from fastapi.responses import StreamingResponse
from geojson_pydantic import FeatureCollection

from app import enums, schemas
from calculations.utils import request_points_project
from calculations.CityMetricsMethods import *
from data.cities_dictionary import cities_model, cities_name

router = APIRouter()

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

@router.get("/cities")
async def get_cities_names():
    return cities_name

@router.post('/pedastrian_walk_traffics/pedastrian_walk_traffics_calculation', 
            response_model=schemas.PedastrianWalkTrafficsCalculationOut, tags=[Tags.trafics_calculation])
def pedastrian_walk_traffics_calculation(query_params: schemas.PedastrianWalkTrafficsCalculationIn):
    city_model = cities_model[query_params.city]
    result = TrafficCalculator(city_model).get_trafic_calculation(query_params.geojson.dict())
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No living houses in the specified area"
        )
    return result


@router.get("/mobility_analysis/isochrones", response_model=schemas.MobilityAnalysisIsochronesOut,
            tags=[Tags.mobility_analysis])
async def mobility_analysis_isochrones(query_params: schemas.MobilityAnalysisIsochronesQueryParams = Depends()):
    if not (query_params.travel_type == enums.MobilityAnalysisIsochronesTravelTypeEnum.PUBLIC_TRANSPORT \
        and query_params.weight_type == enums.MobilityAnalysisIsochronesWeightTypeEnum.TIME) \
        and (query_params.routes == True):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Not implementet yet."
        )
    
    city_model = cities_model[query_params.city]
    request_points = [[query_params.x_from, query_params.y_from]]
    to_crs = cities_model[query_params.city].city_crs
    x_from, y_from = request_points_project(request_points, 4326, to_crs)[0]
    result = AccessibilityIsochrones(city_model).get_accessibility_isochrone(
        travel_type=query_params.travel_type, x_from=x_from, y_from=y_from, 
        weight_type=query_params.weight_type, weight_value=query_params.weight_value, routes=query_params.routes
        )

    return result


@router.get("/visibility_analysis/visibility_analysis", response_model=FeatureCollection,
            tags=[Tags.visibility_analysis])
async def visibility_analisys(query_params: schemas.VisibilityAnalisysQueryParams = Depends()):
    city_model = cities_model[query_params.city]
    request_points = [[query_params.x_from, query_params.y_from]]
    to_crs = cities_model[query_params.city].city_crs
    request_point = request_points_project(request_points, 4326, to_crs)[0]
    return VisibilityAnalysis(city_model).get_visibility_result(request_point, query_params.view_distance)


@router.post("/voronoi/weighted_voronoi_calculation", response_model=schemas.WeightedVoronoiCalculationOut,
             tags=[Tags.weighted_voronoi])
async def wighted_voronoi_calculation(query_params: schemas.WeightedVoronoiCalculationIn):
    city_model = cities_model[query_params.city]
    return WeightedVoronoi(city_model).get_weighted_voronoi_result(query_params.geojson.dict())


@router.post("/blocks_clusterization/get_blocks", response_model=FeatureCollection,
             tags=[Tags.blocks_clusterization])
async def get_blocks_clusterization(query_params: schemas.BlocksClusterizationGetBlocks):
    city_model = cities_model[query_params.city]
    return BlocksClusterization(city_model).get_blocks(
        query_params.service_types, query_params.clusters_number, 
        query_params.area_type, query_params.area_id, query_params.geojson
        )

@router.post("/blocks_clusterization/get_dendrogram",
             responses={
              200: {
                  "content": {"image/png": {}}
              }
          },
             response_class=StreamingResponse,
             tags=[Tags.blocks_clusterization])
async def get_blocks_clusterization_dendrogram(query_params: schemas.BlocksClusterizationGetBlocks):
    city_model = cities_model[query_params.city]
    result = BlocksClusterization(city_model).get_dendrogram(query_params.service_types)
    return StreamingResponse(content=result, media_type="image/png")


@router.post("/services_clusterization/get_clusters_polygons", 
            response_model=schemas.ServicesClusterizationGetClustersPolygonsOut, tags=[Tags.services_clusterization])
async def get_services_clusterization(query_params: schemas.ServicesClusterizationGetClustersPolygonsIn):
    city_model = cities_model[query_params.city]
    result = ServicesClusterization(city_model).get_clusters_polygon(
        query_params.service_types, query_params.area_type, query_params.area_id, query_params.geojson,
        query_params.condition, query_params.condition_value, query_params.n_std
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enough objects to cluster"
        )

    return result


@router.post("/spacematrix/get_indices", response_model=FeatureCollection,
            tags=[Tags.spacematrix])
async def get_spacematrix_indices(query_params: schemas.SpacematrixIn):
    city_model = cities_model[query_params.city]
    geojson = query_params.geojson.dict() if query_params.geojson else None
    return Spacematrix(city_model).get_spacematrix_morph_types(
        query_params.clusters_number, query_params.area_type, query_params.area_id, geojson
        )


@router.get("/diversity/diversity", response_model=schemas.DiversityOut,
            tags=[Tags.diversity])
async def get_diversity(query_params: schemas.DiversityQueryParams = Depends()):  # todo validate service_type?
    city_model = cities_model[query_params.city]
    result = Diversity(city_model).get_diversity(query_params.service_type)
    return result
    
@router.get("/diversity/get_buildings", response_model=FeatureCollection,
            tags=[Tags.diversity])
async def get_buildings_diversity(query_params: schemas.DiversityGetBuildingsQueryParams = Depends()):
    city_model = cities_model[query_params.city]
    result = Diversity(city_model).get_houses(query_params.block_id, query_params.service_type)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The objects (houses and services) with given parametrs are absent"
        )
    return result

@router.get("/diversity/get_info", response_model=schemas.DiversityGetInfoOut,
            tags=[Tags.diversity])
async def get_diversity_info(query_params: schemas.DiversityGetInfoQueryParams = Depends()):
    city_model = cities_model[query_params.city]
    result = Diversity(city_model).get_info(query_params.house_id, query_params.service_type)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The objects (houses and services) with given parametrs are absent"
        )
    return result


# @router.post("/provision/get_provision", response_model=schemas.ProvisionGetProvisionOut,
#              tags=[Tags.provision])
# async def get_provision(user_request: schemas.ProvisionGetProvisionIn):
#     """
#     In user request:
#     required params: service_type, area, provision_type
#     optional params: city, without_services_options, load_options, provision_option
#     :return: dict of FeatureCollections houses and services
#     """
#     result = BCAM.get_provision(**user_request.dict())
#     return result


# @router.post("/provision/get_info", response_model=schemas.ProvisionGetInfoOut,
#              tags=[Tags.provision])
# async def get_provision_info(user_request: schemas.ProvisionGetInfoIn):
#     """
#     In user request:
#     required params: object_type, functional_object_id, service_type, provision_type
#     :return: dict of FeatureCollections of houses, services and isochrone (not for all request)
#     """
#     result = BCAM.get_provision_info(**user_request.dict())
#     return result


# @router.post("/wellbeing/get_wellbeing", response_model=schemas.WellbeingGetWellbeingOut,
#              tags=[Tags.well_being])
# async def get_wellbeing(user_request: schemas.WellbeingGetWellbeingIn):
#     """
#     In user request:
#     required params: provision_type and either living_situation_id or user_service_types
#     :return: dict of FeatureCollections houses and services
#     """
#     result = CMM.get_wellbeing(BCAM, **user_request.dict())
#     return result


# @router.post("/wellbeing/get_wellbeing_info", response_model=schemas.WellbeingGetWellbeingInfoOut,
#              tags=[Tags.well_being])
# async def get_wellbeing_info(user_request: schemas.WellbeingGetWellbeingInfoIn):
#     """
#     In user request:
#     required params: provision_type, object_type, functional_object_id and either living_situation_id or user_service_types
#     :return: dict of FeatureCollections of houses, services, isochrone (not for all request) and service types as json (not for all request)
#     """
#     result = CMM.get_wellbeing_info(BCAM, **user_request.dict())
#     return result

