import pandas as pd
import os
import pickle

from sqlalchemy import create_engine
from .QueryInterface import QueryInterface
from .DataValidation import DataValidation 

class DataQueryInterface(QueryInterface):

    def __init__(self, city_name, city_crs, city_db_id):

        self.city_name = city_name
        self.city_crs = city_crs
        self.city_id = city_db_id
        self.mongo_address = "http://" + os.environ["MONGO"]
        self.engine = create_engine("postgresql://" + os.environ["POSTGRES"])

        self.validation = DataValidation(self.city_name, self.mongo_address)
        # Select with city_id
        self.place_slice = {"place": "city",
                            "place_id": self.city_id}

        self.attr_names = {'MobilityGraph': self._Graphs(), 
                           'Buildings': self._Buildings(), 
                           'Services': self._Services(), 
                           'PublicTransportStops': self._PublicTransportStops(), 
                           'ServiceTypes': self._ServiceTypes(),
                           'RecreationalAreas': self._Recreational_Areas(), 
                           'Blocks': self._Blocks(), 
                           'Municipalities': self._Municipalities(),
                           'AdministrativeUnits': self._Districts(), 
                           'ValueTypes': self._ValueTypes(), 
                           'SocialGroups': self._SocialGroups(), 
                           'SocialGroupsValueTypesLivingSituations': self._SocialGroupsValueTypesLivingSituations(), 
                           'LivingSituationsCityServiceTypes': self._LivingSituationsCityServiceTypes(),
                           'LivingSituations': self._LivingSituations()}
        
    def _Graphs(self):
        MobilityGraph = self.get_graph_for_city(self.city_name, 
                                                "intermodal_graph", node_type=int)
        self.validation.validate_graphml("MobilityGraph", MobilityGraph)
        yield MobilityGraph

    def _Buildings(self):
        buildings_columns = ["building_id as id", "building_area as basement_area", "is_living", "living_area", 
                            "population_balanced as population", "storeys_count", 
                            "building_year", "central_heating", "central_hotwater", "central_electro",
                            "central_gas", "failure as is_emergency", "project_type", 
                            "functional_object_id", "address", "administrative_unit_id", "municipality_id", 
                            "block_id", "geometry"]

        Buildings = self.get_buildings(buildings_columns, 
                                       self.place_slice).to_crs(self.city_crs)
        self.validation.validate_df("Buildings", Buildings, "geojson")
        yield Buildings

    def _Services(self):
        service_columns = ["building_id", "functional_object_id as id", "city_service_type",
                           "city_service_type_id", "city_service_type_code as service_code", "service_name",
                           "address", "capacity", "block_id", "administrative_unit_id", "municipality_id",
                           "center as geometry"]
             
        Services = self.get_services(service_columns, 
                                     place_slice = self.place_slice)
        self.validation.validate_df("Services", Services, "geojson")
        yield Services

    def _PublicTransportStops(self):
        Services = next(self._Services())
        PublicTransportStops = self.Services[Services["service_code"] == "stops"]
        self.validation.validate_df("PublicTransportStops", PublicTransportStops, "geojson")
        yield PublicTransportStops

    def _ServiceTypes(self):
        ServiceTypes = pd.read_sql_table("city_service_types", 
                                         con = self.engine, 
                                         columns = ["id", "code", "name", 
                                                    "public_transport_time_normative", 
                                                    "walking_radius_normative"])
        self.validation.validate_df("ServiceTypes", ServiceTypes, "json")

        yield ServiceTypes
    
    def _Recreational_Areas(self):
        rec_areas_columns = ["functional_object_id as id", "city_service_type", 
                             "geometry", "city_service_type_id", 
                             "cast(functional_object_properties->>'ndvi' as double precision) as vegetation_index",
                             "city_service_type_code as service_code", "service_name",
                             "address", "capacity", "block_id", "administrative_unit_id", "municipality_id"]
                            
        RecreationalAreas = self.get_services(rec_areas_columns, 
                                              place_slice = self.place_slice, 
                                              equal_slice = {"column": "city_service_type_code",
                                                             "value": "recreational_areas"})
        yield RecreationalAreas

    def _Blocks(self):
        blocks_column = ["id", "area", 
                         "municipality_id", 
                         "administrative_unit_id", 
                         "geometry"]
        Blocks = self.get_territorial_units("blocks", 
                                            blocks_column, 
                                            place_slice = self.place_slice)
        self.validation.validate_df("Blocks", Blocks, "geojson")
        yield Blocks


    def _Municipalities(self):
        Municipalities = self.get_territorial_units("municipalities", 
                                                    ["id", "geometry"],
                                                    place_slice = self.place_slice)
        self.validation.validate_df("Municipalities", self.Municipalities, "geojson")
        yield Municipalities

    def _Districts(self):
        AdministrativeUnits = self.get_territorial_units("administrative_units",
                                                         ["id", "geometry", "name"],
                                                         place_slice = self.place_slice)
        yield AdministrativeUnits

    def _ValueTypes(self):
        ValueTypes = pd.read_sql('''SELECT vt.id AS value_type_id, vt.name AS value_type, vg.id AS value_group_id, vg.name AS value_group
                                    FROM maintenance.value_types vt
                                    JOIN maintenance.value_groups vg ON vt.group_id = vg.id
                                    ORDER BY vg.id, vt.id
                                    ''', con = self.engine)
        yield ValueTypes               
        
    def _SocialGroups(self):
        SocialGroups = pd.read_sql('''SELECT id as social_groups_id, name as social_groups_name
                                      FROM public.social_groups
                                      ''', con = self.engine)
        yield SocialGroups

    def _SocialGroupsValueTypesLivingSituations(self):
        SocialGroupsValueTypesLivingSituations = pd.read_sql('''SELECT *
                                                                FROM maintenance.social_groups_value_types_living_situations
                                                                ''', con = self.engine)
        yield SocialGroupsValueTypesLivingSituations
    
    def _LivingSituationsCityServiceTypes(self):
        LivingSituationsCityServiceTypes = pd.read_sql('''SELECT living_situation_id, city_service_type_id
                                                          FROM maintenance.living_situations_city_service_types
                                                          ''', con = self.engine).drop_duplicates()
        yield LivingSituationsCityServiceTypes
    def _LivingSituations(self):

        LivingSituations = pd.read_sql('''SELECT id as living_situation_id, name as living_situations_name
                                          FROM public.living_situations
                                          ''', con = self.engine)
        yield LivingSituations
