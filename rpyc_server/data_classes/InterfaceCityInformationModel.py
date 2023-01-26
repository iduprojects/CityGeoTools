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
        place_slice = {"place": "city", "place_id": self.city_id}
        
        # Graphs
        self.MobilityGraph = self.get_graph_for_city(city_name, "intermodal_graph", node_type=int)
        self.validation.validate_graphml("MobilityGraph", self.MobilityGraph)
        self.MobilityGraph = pickle.dumps(self.MobilityGraph)


        # Buildings
        buildings_columns = ["building_id as id", "building_area as basement_area", "is_living", "living_area", 
                            "population_balanced as population", "storeys_count", 
                            "building_year", "central_heating", "central_hotwater", "central_electro",
                            "central_gas", "failure as is_emergency", "project_type", 
                            "functional_object_id", "address", "administrative_unit_id", "municipality_id", 
                            "block_id", "geometry"]

        self.Buildings = self.get_buildings(buildings_columns, place_slice).to_crs(self.city_crs)
        self.validation.validate_df("Buildings", self.Buildings, "geojson")
        self.Buildings = pickle.dumps(self.Buildings.to_crs(self.city_crs))


        # Services
        service_columns = ["building_id", "functional_object_id as id", "city_service_type",
                            "city_service_type_id", "city_service_type_code as service_code", "service_name",
                            "address", "capacity", "block_id", "administrative_unit_id", "municipality_id",
                            "center as geometry"]
                            
        self.Services = self.get_services(service_columns, place_slice=place_slice)
        self.validation.validate_df("Services", self.Services, "geojson")

        self.PublicTransportStops = self.Services[self.Services["service_code"] == "stops"]
        self.validation.validate_df("PublicTransportStops", self.PublicTransportStops, "geojson")

        self.ServiceTypes = pd.read_sql_table(
            "city_service_types", con=self.engine, 
            columns=["id", "code", "name", "public_transport_time_normative", "walking_radius_normative"])
        self.validation.validate_df("ServiceTypes", self.ServiceTypes, "json")

        self.Services = pickle.dumps(self.Services.to_crs(self.city_crs))
        self.ServiceTypes = pickle.dumps(self.ServiceTypes)
        self.PublicTransportStops = pickle.dumps(self.PublicTransportStops.to_crs(self.city_crs))

        # Recreational Areas
        rec_areas_columns = ["functional_object_id as id", "city_service_type", "geometry", "city_service_type_id", 
                            "cast(functional_object_properties->>'ndvi' as double precision) as vegetation_index",
                            "city_service_type_code as service_code", "service_name",
                            "address", "capacity", "block_id", "administrative_unit_id", "municipality_id"]
                            
        self.RecreationalAreas = self.get_services(
            rec_areas_columns, place_slice=place_slice, 
            equal_slice={"column": "city_service_type_code", "value": "recreational_areas"}
            )
        self.RecreationalAreas = pickle.dumps(self.RecreationalAreas.to_crs(self.city_crs))

        # Blocks
        blocks_column = ["id", "area", "municipality_id", "administrative_unit_id", "geometry"]
        self.Blocks = self.get_territorial_units("blocks", blocks_column, place_slice=place_slice)
        self.validation.validate_df("Blocks", self.Blocks, "geojson")
        self.Blocks = pickle.dumps(self.Blocks.to_crs(self.city_crs))


        # Municipalities
        self.Municipalities = self.get_territorial_units("municipalities", ["id", "geometry"], place_slice=place_slice)
        self.validation.validate_df("Municipalities", self.Municipalities, "geojson")
        self.Municipalities = pickle.dumps(self.Municipalities.to_crs(self.city_crs))

        # Districts
        self.AdministrativeUnits = self.get_territorial_units(
            "administrative_units", ["id", "geometry", "name"], place_slice=place_slice
        )
        self.AdministrativeUnits = pickle.dumps(self.AdministrativeUnits.to_crs(self.city_crs))

        self.ValueTypes = pd.read_sql('''SELECT vt.id AS value_type_id, vt.name AS value_type, vg.id AS value_group_id, vg.name AS value_group
                                         FROM maintenance.value_types vt
                                         JOIN maintenance.value_groups vg ON vt.group_id = vg.id
                                         ORDER BY vg.id, vt.id
                                         ''', con = self.engine)
        self.ValueTypes = pickle.dumps(self.ValueTypes)               
        
        self.SocialGroups = pd.read_sql('''SELECT id as social_groups_id, name as social_groups_name
                                            FROM public.social_groups
                                            ''', con = self.engine)
        self.SocialGroups = pickle.dumps(self.SocialGroups)   

        self.SocialGroupsValueTypesLivingSituations = pd.read_sql('''SELECT *
                                                                          FROM maintenance.social_groups_value_types_living_situations
                                                                          ''', con = self.engine)
        self.SocialGroupsValueTypesLivingSituations = pickle.dumps(self.SocialGroupsValueTypesLivingSituations)   

        self.LivingSituationsCityServiceTypes = pd.read_sql('''SELECT living_situation_id, city_service_type_id
                                                                   FROM maintenance.living_situations_city_service_types
                                                                   ''', con = self.engine).drop_duplicates()
        self.LivingSituationsCityServiceTypes = pickle.dumps(self.LivingSituationsCityServiceTypes) 

        self.LivingSituations = pd.read_sql('''SELECT id as living_situation_id, name as living_situations_name
                                                FROM public.living_situations
                                                ''', con = self.engine)
        self.LivingSituations = pickle.dumps(self.LivingSituations)   

        self.readiness = all(self.validation.__dict__.values())
        self.readiness = pickle.dumps(self.readiness)