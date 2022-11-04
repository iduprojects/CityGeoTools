from typing import Union, Optional, Dict

from fastapi import Query
from pydantic import BaseModel, validator, Field, conint, conlist, confloat, root_validator
from geojson_pydantic import LineString, FeatureCollection, Polygon
from geojson_pydantic.geometries import Geometry
from geojson_pydantic.features import Props

from . import enums


def _check_latitude_epsg_4326(lat):
    """ Check latitude for EPSG:4326. """
    if not -90 <= lat <= 90:
        raise ValueError("Не верное значение широты для проекции EPSG:4326")

    return lat


def _check_longitude_epsg_4326(lon):
    """ Check longitude for EPSG:4326. """
    if not -180 <= lon <= 180:
        raise ValueError("Не верное значение долготы для проекции EPSG:4326")

    return lon

class FeatureCollectionWithCRS(FeatureCollection):
    crs: dict

# /pedastrian_walk_traffics/pedastrian_walk_traffics_calculation
class PedastrianWalkTrafficsCalculationIn(BaseModel):
    city: enums.CitiesEnum
    geojson: FeatureCollectionWithCRS

    class Config:
        schema_extra = {
            "example": {
                "city": "saint-petersburg",
                "geojson": {
                "type": "FeatureCollection",
                "name": "test_area",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": [
                    {"type": "Feature", "properties": {},
                     "geometry": {"type": "Polygon",
                                  "coordinates": [[[30.253654233376736, 59.952160447703385],
                                                   [30.255163054210676, 59.95547985353806],
                                                   [30.26592597615947, 59.954205738167175],
                                                   [30.268843029771755, 59.95102044973996],
                                                   [30.266428916437448, 59.94884104186871],
                                                   [30.258013049119235, 59.94833810159073],
                                                   [30.253654233376736, 59.952160447703385]]]}}]}}
        }

class PedastrianWalkTrafficsCalculationOut(BaseModel):
    buildings: FeatureCollection
    stops: FeatureCollection
    routes: FeatureCollection


# /mobility_analysis/isochrones
class MobilityAnalysisIsochronesQueryParams:
    def __init__(self,
                 city: enums.CitiesEnum,
                 travel_type: enums.MobilityAnalysisIsochronesTravelTypeEnum,
                 weight_type: enums.MobilityAnalysisIsochronesWeightTypeEnum,
                 weight_value: conint(ge=1) = Query(..., example=10),
                 x_from: float = Query(..., example=59.94288),
                 y_from: float = Query(..., example=30.31413),
                 routes: bool = False
                 ):
        self.city = city
        self.travel_type = travel_type
        self.weight_type = weight_type
        self.weight_value = weight_value
        self.x_from = _check_latitude_epsg_4326(x_from)
        self.y_from = _check_longitude_epsg_4326(y_from)
        self.routes = routes

    class Config:
        schema_extra = {
            'example': [
                {
                    'city': 'saint-petersburg',
                    'travel_type': 'public_transport',
                    'weight_type': 'time',
                    'weight_value': 15,
                    'x_from': 59.8059,
                    'y_from': 30.4267,
                    'routes': True
                }
            ]
        }


MobilityAnalysisIsochronesGeometry = Geometry
class MobilityAnalysisIsochronesProperties(BaseModel):
    travel_type: enums.MobilityAnalysisIsochronesTravelTypeLabelEnum
    weight_type: enums.MobilityAnalysisIsochronesWeightTypeEnum
    weight_value: conint(ge=1)

class MobilityAnalysisIsochronesOut(BaseModel):
    isochrone: FeatureCollection[
    MobilityAnalysisIsochronesGeometry,
    MobilityAnalysisIsochronesProperties
]
    stops: Optional[FeatureCollection]
    routes: Optional[FeatureCollection]

# /Visibility_analysis/Visibility_analysis
class VisibilityAnalisysQueryParams:
    def __init__(self,
                 city: enums.CitiesEnum,
                 x_from: float = Query(..., example=59.944548),
                 y_from: float = Query(..., example=30.304617),
                 view_distance: int = Query(..., example=700),
                 ):
        self.x_from = x_from
        self.y_from = y_from
        self.city = city
        self.view_distance = view_distance

# /voronoi/Weighted_voronoi_calculation
class WeightedVoronoiCalculationIn(BaseModel):
    city: enums.CitiesEnum
    geojson: FeatureCollectionWithCRS

    class Config:
        schema_extra = {
            "example": {
                "city": "saint-petersburg",
                "geojson": {
                    "type": "FeatureCollection", "name": "test",
                    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
                    "features": [{"type": "Feature", "properties": {"weight": 5.330623275960242},
                                  "geometry": {"type": "Point",
                                               "coordinates": [3365424.7412466537, 8388892.177795848]}},
                                 {"type": "Feature", "properties": {"weight": 6.51134526170844},
                                  "geometry": {"type": "Point",
                                               "coordinates": [3366784.3993621334, 8390015.740133371]}},
                                 {"type": "Feature", "properties": {"weight": 4.7412358799581495},
                                  "geometry": {"type": "Point",
                                               "coordinates": [3366811.6122107455, 8384317.056265167]}}]
                }
            }
        }

class WeightedVoronoiCalculationOut(BaseModel):
    voronoi_polygons: FeatureCollection
    deficit_zones: FeatureCollection

ServicesList = conlist(str, min_items=1, unique_items=True)

# /blocks_clusterization/get_blocks
class BlocksClusterizationGetBlocks(BaseModel):
    city: enums.CitiesEnum
    service_types: ServicesList # todo service types as enumerate
    clusters_number: Optional[int]
    area_type: Optional[enums.TerritorialEnum]
    area_id: Optional[int]
    geojson: Optional[FeatureCollectionWithCRS]

    class Config:
        schema_extra = {
            "example": {
                "city": "saint-petersburg",
                "clusters_number": None,
                "service_types": [
                        "garbage_containers",
                        "bakeries",
                        "dentists",
                        "parkings",
                        "pet_shops"
                    ],
                "area_type": "administrative_unit",
                "area_id": 61
                }
            }

class ServicesClusterizationGetClustersPolygonsIn(BaseModel):
    city: enums.CitiesEnum
    service_types: ServicesList
    area_type: Optional[enums.TerritorialEnum]
    area_id: Optional[int]
    geojson: Optional[FeatureCollectionWithCRS]
    condition: enums.ClusterizationConditionsEnum
    condition_value: Optional[int]
    n_std: int = 2

    @root_validator
    def validate_condition_value(cls, values):
        if values["condition"] == "distance":
            values["condition_value"] = 4000
            return values
        else:
            values["condition_value"] = 10
            return values

    class Config:
        schema_extra = {
            "example": {
                "city": "saint-petersburg",
                "service_types": [
                        "schools",
                        "kindergartens",
                        "colleges",
                        "universities"
                    ],
                "condition": "distance",
                "condition_value": 4000,
                "n_std": 2,
                "area_type": "administrative_unit",
                "area_id": 61
                }
            }

class ServicesClusterizationGetClustersPolygonsOut(BaseModel):
    polygons: FeatureCollection
    services: FeatureCollection

class SpacematrixIn(BaseModel):
    city: enums.CitiesEnum
    clusters_number: int = 11
    area_type: Optional[enums.TerritorialEnum]
    area_id: Optional[int]
    geojson: Optional[FeatureCollectionWithCRS]

    class Config:
        schema_extra = {
            "example": {
                "city": "saint-petersburg",
                "clusters_number": 11,
                "area_type": "administrative_unit",
                "area_id": 61
                }
            }


class DiversityQueryParams:
    def __init__(self,
                 city: enums.CitiesEnum,
                 service_type: str = Query(..., example="cafes")
                 ):
        self.city = city
        self.service_type = service_type


class DiversityOut(BaseModel):
    municipalities: FeatureCollection
    blocks: FeatureCollection


class DiversityGetInfoQueryParams:
    def __init__(self,
                 city: enums.CitiesEnum,
                 house_id: int = Query(..., example=45793),
                 service_type: str = Query(..., example="cafes")
                 ):
        self.city = city
        self.house_id = house_id
        self.service_type = service_type

class DiversityGetInfoOut(BaseModel):
    house: FeatureCollection
    services: FeatureCollection
    isochrone: FeatureCollection


class DiversityGetBuildingsQueryParams:
    def __init__(self,
                 city: enums.CitiesEnum,
                 block_id: int = Query(..., example=488),
                 service_type: str = Query(..., example="cafes")
                 ):
        self.city = city
        self.block_id = block_id
        self.service_type = service_type


ProvisionsDestinationMatrix = dict[str, list[dict]]  # матрица назначений


class ProvisionInBase(BaseModel):
    """Базовый класс схемы входных параметров для обеспеченности. """
    city: str
    service_types: conlist(str, min_items=1)
    valuation_type: str
    year: int
    user_selection_zone: Optional[Polygon] = None
    service_impotancy: Optional[list] = None  #FIXME

class ProvisionGetProvisionIn(ProvisionInBase):
    class Config:
        schema_extra = {
            "example": {
                "city": "saint-petersburg",
<<<<<<< HEAD
                "service_types": ["kindergartens"],
=======
                "service_type": "kindergartens",
>>>>>>> master
                "valuation_type": "normative",
                "year": 2022,
            }
        }


class ProvisionRecalculateProvisionsIn(ProvisionInBase):
    user_provisions: ProvisionsDestinationMatrix
    user_changes_buildings: Optional[dict] = None
    user_changes_services: Optional[dict] = None


class ProvisionOutBase(BaseModel):
    houses: FeatureCollection
    services: FeatureCollection
    provisions: ProvisionsDestinationMatrix


class ProvisionGetProvisionOut(ProvisionOutBase):
    ...


class ProvisionGetInfoIn(BaseModel):
    object_type: enums.ProvisionGetInfoObjectTypeEnum
    functional_object_id: int

    class Config:
        schema_extra = {
            "example": {
                "object_type": "house",
                "functional_object_id": 112474,
                "service_type": "bars",
                "provision_type": "normative"
            }
        }


class ProvisionGetInfoOut(ProvisionOutBase):
    isochrone: Optional[FeatureCollection]


class WellbeingInBase(BaseModel):
    provision_type: enums.ProvisionTypeEnum
    living_situation_id: Optional[int]
    user_service_types: Optional[Dict[str, int]]


class WellbeingGetWellbeingIn(WellbeingInBase):
    area: Union[dict[str, int], FeatureCollection]
    wellbeing_option: conlist(int, min_items=2, max_items=2)

    @validator("wellbeing_option")
    def check_user_price_preferences_order(cls, v):
        min_, max_ = v
        if not min_ <= max_:
            raise ValueError("Left bound must be less or equal than right bound.")

    class Config:
        schema_extra = {
            "example": {
                "living_situation_id": 3,
                "area": {
                    "district": 61
                },
                "provision_type": "normative",
                "wellbeing_option": [0, 0.4]
            }
        }


class WellbeingGetWellbeingOut(ProvisionOutBase):
    ...


class WellbeingGetWellbeingInfoIn(WellbeingInBase):
    object_type: enums.ProvisionGetInfoObjectTypeEnum
    functional_object_id: int

    class Config:
        schema_extra = {
            "example": {
                "object_type": "house",
                "functional_object_id": 112225,
                "provision_type": "calculated",
                "user_service_types": {
                    "bakeries": 0.5,
                    "banks": 0.5
                }
            }
        }


class WellbeingGetWellbeingInfoOut(ProvisionOutBase):
    isochrone: Optional[FeatureCollection]


class CollocationMatrixQueryParams:
    def __init__(self,
                 city: enums.CitiesEnum,
                 ):
        self.city = city
