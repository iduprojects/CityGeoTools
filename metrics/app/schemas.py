from typing import Union, Optional, Dict

from fastapi import Query
from pydantic import BaseModel, validator, Field, conint, conlist, confloat, root_validator
from geojson_pydantic import LineString, FeatureCollection
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

# /pedastrian_walk_traffics/pedastrian_walk_traffics_calculation
class PedastrianWalkTrafficsCalculationIn(BaseModel):
    class FeatureCollectionWithCRS(FeatureCollection):
        crs: dict
    city: enums.CitiesEnum
    geojson: FeatureCollectionWithCRS

    class Config:
        schema_extra = {
            "example": {
                "city": "Saint_Petersburg",
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
                 ):
        self.city = city
        self.travel_type = travel_type
        self.weight_type = weight_type
        self.weight_value = weight_value
        self.x_from = _check_latitude_epsg_4326(x_from)
        self.y_from = _check_longitude_epsg_4326(y_from)

    class Config:
        schema_extra = {
            'example': [
                {
                    'city': 'Saint_Petersburg',
                    'travel_type': 'public_transport',
                    'weight_type': 'time',
                    'weight_value': 10,
                    'x_from': 59.8059,
                    'y_from': 30.4267,
                }
            ]
        }


class MobilityAnalysisIsochronesProperties(BaseModel):
    travel_type: enums.MobilityAnalysisIsochronesTravelTypeLabelEnum
    weight_type: enums.MobilityAnalysisIsochronesWeightTypeEnum


MobilityAnalysisIsochronesGeometry = Geometry
MobilityAnalysisIsochronesOut = FeatureCollection[
    MobilityAnalysisIsochronesGeometry,
    MobilityAnalysisIsochronesProperties
]

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
    class FeatureCollectionWithCRS(FeatureCollection):
        crs: dict

    city: enums.CitiesEnum
    geojson: FeatureCollectionWithCRS

    class Config:
        schema_extra = {
            "example": {
                "city": "Saint_Petersburg",
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
    service_types: list # todo service types as enumerate
    clusters_number: Optional[int]

    class Config:
        schema_extra = {
            "example": {
                "city": "Saint_Petersburg",
                "clusters_number": None,
                "service_types": [
                        "garbage_containers",
                        "bakeries",
                        "dentists",
                        "parkings",
                        "pet_shops"
                    ]
                }
            }


class GetServices(BaseModel):
    class Param(BaseModel):  # todo move to outside class
        service_types: Optional[ServicesList]
        area: Optional[dict]  # todo area_type and area_id

    city: enums.CitiesEnum
    param: Param


class BlocksClusterizationGetServices(GetServices):
    ...  # todo block area_type and area_id Optional[int]

    class Config:
        schema_extra = {
            "example": {
                "city": "Saint_Petersburg",
                "param": {
                    "service_types": [
                        "bars",
                        "cafes",
                        "colleges",
                        "restaurants"
                    ],
                    "area": {
                        "block": 856
                    }
                }
            }
        }


class ServicesClusterizationGetServices(GetServices):
    ...

    class Config:
        schema_extra = {
            "example": {
                "city": "Saint_Petersburg",
                "param": {
                    "clusters_number": "default",
                    "service_types": [
                        "garbage_containers",
                        "bakeries",
                        "dentists",
                        "parkings",
                        "pet_shops"
                    ]
                }
            }
        }


class ServicesClusterizationGetClustersPolygonsIn(BaseModel):
    class Param(BaseModel):  # todo move to outside class
        service_types: ServicesList
        area: Optional[dict]  # todo mo or district area_type and area_id Optional[int]
        condition: dict = {
            "default": "default"}  # todo condition: str = Field("distance", regex=r"(distance|maxclust)")
        n_std: Optional[int]

    city: enums.CitiesEnum
    param: Param

    class Config:
        schema_extra = {
            "example": {
                "city": "Saint_Petersburg",
                "param": {
                    "service_types": [
                        "schools",
                        "kindergartens",
                        "colleges",
                        "universities"
                    ],
                    "condition": {
                        "distance": 4000
                    },
                    "n_std": 2
                }
            }
        }


class ServiceLocationServiceLocationIn(BaseModel):
    user_service_choice: str
    user_unit_square_min: int
    user_unit_square_max: int

    @root_validator
    def user_unit_square_min_less_or_equal_max(cls, values):
        min_, max_ = values.get("user_unit_square_min"), values.get("user_unit_square_max")
        if not min_ <= max_:
            raise ValueError(
                "user_unit_square_min must be less or equal user_unit_square_max"
            )

        return values

    class Config:
        schema_extra = {
            "example": {
                "user_service_choice": "Кафе/столовая",
                "user_unit_square_min": 10,
                "user_unit_square_max": 50
            }
        }


class ServiceLocationServiceLocationOut(BaseModel):
    municipalities: FeatureCollection
    blocks: FeatureCollection
    rent_ads: FeatureCollection


class DiversityDiversityQueryParams:
    def __init__(self,
                 service_type: str = Query(..., example="cafes")
                 ):
        self.service_type = service_type


class DiversityDiversityOut(BaseModel):
    municipalities: FeatureCollection
    blocks: Dict[int, FeatureCollection]


class ProvisionInBase(BaseModel):
    service_type: Union[list, str]  # todo only list or Union[list, str]?
    provision_type: enums.ProvisionTypeEnum

    is_weighted: bool = False  # todo is_weighted and service_coef inheritance or Mixin
    service_coef: Optional[
        Dict[str, int]]  # todo check sum coef and all service in service_type and only if is_weighted


class ProvisionGetProvisionIn(ProvisionInBase):
    area: Union[dict[str, int], FeatureCollection]
    city: enums.CitiesEnum = enums.CitiesEnum.SAINT_PETERSBURG
    without_services_options: bool = False
    load_option: enums.ProvisionLoadOptionEnum = enums.ProvisionLoadOptionEnum.ALL_SERVICES
    provision_option: enums.ProvisionOptionEnum = enums.ProvisionOptionEnum.ALL_HOUSES

    class Config:
        schema_extra = {
            "example": {
                "service_type": [
                    "flower_stores",
                    "jewelry_stores"
                ],
                "area": {
                    "mo": 67
                },
                "provision_type": "normative",
                "is_weighted": "True",
                "service_coef": {
                    "flower_stores": 0.6,
                    "jewelry_stores": 0.4
                },
                "provision_option": 0
            }
        }


class ProvisionOutBase(BaseModel):
    houses: FeatureCollection
    services: FeatureCollection


class ProvisionGetProvisionOut(ProvisionOutBase):
    ...


class ProvisionGetInfoIn(ProvisionInBase):
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
