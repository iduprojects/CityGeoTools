from enum import auto
import faulthandler

from fastapi import APIRouter, HTTPException, status, Body, Depends
from fastapi.responses import StreamingResponse
from geojson_pydantic import FeatureCollection

from app import enums, schemas
from calculations.utils import request_points_project
from calculations.CityMetricsMethods import *
from calculations import errors
from data.city_models import city_models, city_names

router = APIRouter()
faulthandler.enable()

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
    collocation_matrix = auto()
    city_context = auto()


@router.get("/")
async def read_root():
    return {"Hello": "World"}

@router.get("/cities")
async def get_cities_names():
    return city_names

@router.post(
    '/pedastrian_walk_traffics/pedastrian_walk_traffics_calculation',
    response_model=schemas.PedastrianWalkTrafficsCalculationOut, tags=[Tags.trafics_calculation]
)
def pedastrian_walk_traffics_calculation(query_params: schemas.PedastrianWalkTrafficsCalculationIn):
    city_model = city_models[query_params.city]
    try:
        result = TrafficCalculator(city_model).get_trafic_calculation(query_params.geojson.dict())
        return result
    except errors.TerritorialSelectError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No living houses in the specified area"
        )


@router.get(
    "/visibility_analysis/visibility_analysis",
    response_model=FeatureCollection, tags=[Tags.visibility_analysis]
)
async def visibility_analysis(query_params: schemas.VisibilityAnalisysQueryParams = Depends()):
    city_model = city_models[query_params.city]
    request_points = [[query_params.x_from, query_params.y_from]]
    to_crs = city_models[query_params.city].city_crs
    request_point = request_points_project(request_points, 4326, to_crs)[0]
    return VisibilityAnalysis(city_model).get_visibility_result(request_point, query_params.view_distance)


@router.post(
    "/voronoi/weighted_voronoi_calculation",
    response_model=schemas.WeightedVoronoiCalculationOut, tags=[Tags.weighted_voronoi]
)
async def wighted_voronoi_calculation(query_params: schemas.WeightedVoronoiCalculationIn):
    city_model = city_models[query_params.city]
    return WeightedVoronoi(city_model).get_weighted_voronoi_result(query_params.geojson.dict())


@router.post(
    "/blocks_clusterization/get_blocks",
    response_model=FeatureCollection, tags=[Tags.blocks_clusterization]
)
async def get_blocks_clusterization(query_params: schemas.BlocksClusterizationGetBlocks):
    city_model = city_models[query_params.city]
    geojson = query_params.geojson.dict() if query_params.geojson else None
    return BlocksClusterization(city_model).get_blocks(
        query_params.service_types, query_params.clusters_number, 
        query_params.area_type, query_params.area_id, geojson
        )


@router.post(
    "/blocks_clusterization/get_dendrogram",
    responses={
        200: {
            "content": {"image/png": {}}
        }
    },
    response_class=StreamingResponse, tags=[Tags.blocks_clusterization]
)
async def get_blocks_clusterization_dendrogram(query_params: schemas.BlocksClusterizationGetBlocks):
    city_model = city_models[query_params.city]
    result = BlocksClusterization(city_model).get_dendrogram(query_params.service_types)
    return StreamingResponse(content=result, media_type="image/png")


@router.post(
    "/services_clusterization/get_clusters_polygons",
    response_model=schemas.ServicesClusterizationGetClustersPolygonsOut, tags=[Tags.services_clusterization])
async def get_services_clusterization(query_params: schemas.ServicesClusterizationGetClustersPolygonsIn):
    city_model = city_models[query_params.city]
    geojson = query_params.geojson.dict() if query_params.geojson else None

    try:
        result = ServicesClusterization(city_model).get_clusters_polygon(
            query_params.service_types, query_params.area_type, query_params.area_id, geojson,
            query_params.condition, query_params.condition_value, query_params.n_std
            )
        return result
    except errors.TerritorialSelectError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/spacematrix/get_indices",
    response_model=FeatureCollection, tags=[Tags.spacematrix]
)
async def get_spacematrix_indices(query_params: schemas.SpacematrixIn):
    city_model = city_models[query_params.city]
    geojson = query_params.geojson.dict() if query_params.geojson else None
    try:
        return Spacematrix(city_model).get_morphotypes(
            query_params.clusters_number, query_params.area_type, query_params.area_id, geojson
            )
    except SelectedValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )


@router.get(
    "/mobility_analysis/isochrones",
    response_model=schemas.MobilityAnalysisIsochronesOut, tags=[Tags.mobility_analysis]
)
async def mobility_analysis_isochrones(query_params: schemas.MobilityAnalysisIsochronesQueryParams = Depends()):
    city_model = city_models[query_params.city]
    request_points = [[query_params.x_from, query_params.y_from]]
    to_crs = city_models[query_params.city].city_crs
    x_from, y_from = request_points_project(request_points, 4326, to_crs)[0]
    try:
        result = AccessibilityIsochrones(city_model).get_accessibility_isochrone(
            travel_type=query_params.travel_type, x_from=x_from, y_from=y_from,
            weight_type=query_params.weight_type, weight_value=query_params.weight_value, routes=query_params.routes
        )

        return result
    except errors.ImplementationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )


@router.get("/diversity/diversity", response_model=schemas.DiversityOut,
            tags=[Tags.diversity])
async def get_diversity(query_params: schemas.DiversityQueryParams = Depends()):  # todo validate service_type?
    city_model = city_models[query_params.city]
    result = Diversity(city_model).get_diversity(query_params.service_type)
    return result

@router.get("/diversity/get_buildings", response_model=FeatureCollection,
            tags=[Tags.diversity])
async def get_buildings_diversity(query_params: schemas.DiversityGetBuildingsQueryParams = Depends()):
    city_model = city_models[query_params.city]
    try:
        result = Diversity(city_model).get_houses(query_params.block_id, query_params.service_type)
        return result
    except (errors.TerritorialSelectError, errors.SelectedValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )


@router.get("/diversity/get_info", response_model=schemas.DiversityGetInfoOut,
            tags=[Tags.diversity])
async def get_diversity_info(query_params: schemas.DiversityGetInfoQueryParams = Depends()):
    city_model = city_models[query_params.city]
    try:
        result = Diversity(city_model).get_info(query_params.house_id, query_params.service_type)
        return result
    except (errors.TerritorialSelectError, errors.SelectedValueError) as e:

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )


@router.post("/provision/get_provision", response_model=schemas.ProvisionGetProvisionOut,
             tags=[Tags.provision])
async def get_provision(
        user_request: schemas.ProvisionGetProvisionIn,
):
    city_model = city_models[user_request.city]
    result = City_Provisions(
        city_model, user_request.service_types,
        user_request.valuation_type, user_request.year,
        user_changes_buildings=None, user_changes_services=None,
        user_provisions=None, user_selection_zone=user_request.user_selection_zone,
        service_impotancy=user_request.service_impotancy,
        return_jsons=True,
    ).get_provisions()
    return result


@router.post("/provision/recalculate_provisions", response_model=schemas.ProvisionGetProvisionOut,
             tags=[Tags.provision])
async def recalculate_provisions(
        user_request: schemas.ProvisionRecalculateProvisionsIn,
):
    city_model = city_models[user_request.city]
    result = City_Provisions(
        city_model, user_request.service_types,
        user_request.valuation_type, user_request.year,
        user_changes_buildings=user_request.user_changes_buildings, user_changes_services=user_request.user_changes_services,
        user_provisions=user_request.user_provisions, user_selection_zone=user_request.user_selection_zone,
        service_impotancy=user_request.service_impotancy
    ).recalculate_provisions()
    return result


@router.get(
    "/collocation_matrix/collocation_matrix",
    response_model=dict[str, dict[str, Optional[float]]], tags=[Tags.collocation_matrix]
)
async def get_collocation_matrix(query_params: schemas.CollocationMatrixQueryParams = Depends()):
    city_model = city_models[query_params.city]
    return CollocationMatrix(city_model).get_collocation_matrix()


@router.post(
    "/city_context/get_context",
    response_model=schemas.CityContextGetContextOut, tags=[Tags.city_context],
)
def city_context_get_context(
        user_request: schemas.CityContextGetContextIn
):
    city_model = city_models[user_request.city]
    return City_context(
        city_model, service_types=user_request.service_types,
        valuation_type=user_request.valuation_type,
        year=user_request.year,
        user_context_zone=user_request.user_selection_zone
    ).get_context()
