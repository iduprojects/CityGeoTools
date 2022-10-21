import pandas as pd
import os
import pickle
import datetime

from sqlalchemy import create_engine
from .QueryInterface import QueryInterface


class DataQueryInterface(QueryInterface):

    def __init__(self, city_name, city_crs, city_db_id):

        self.city_name = city_name
        self.city_crs = city_crs
        self.city_id = city_db_id

        self.mongo_address = "http://" + os.environ["MONGO"]
        self.engine = create_engine("postgresql://" + os.environ["POSTGRES"])
        
        # Select with city_id
        place_slice = {"place": "city", "place_id": self.city_id}
        
        # Graphs
        self.walk_graph = self.get_graph_for_city(city_name, "walk_graph", node_type=int)
        self.walk_graph = pickle.dumps(self.walk_graph)
        print(self.city_name, datetime.datetime.now(),'walk_graph')

        self.drive_graph = self.get_graph_for_city(city_name, "drive_graph", node_type=int)
        self.drive_graph = pickle.dumps(self.drive_graph)
        print(self.city_name, datetime.datetime.now(),'drive_graph')

        self.public_transport_graph = self.get_graph_for_city(city_name, "multimodal_graph", node_type=str)
        self.set_xy_attributes(self.public_transport_graph)
        self.public_transport_graph = pickle.dumps(self.public_transport_graph)
        print(self.city_name, datetime.datetime.now(),'public_transport_graph')


        # Buildings
        buildings_columns = ["building_id as id", "building_area as basement_area", "is_living", "living_area", 
                            "population_balanced as population", "storeys_count", "functional_object_id", "address",
                             "administrative_unit_id", "municipality_id", "block_id", "geometry"]

        self.Buildings = self.get_buildings(buildings_columns, place_slice).to_crs(self.city_crs)
        self.Buildings = self.Buildings[
            (self.Buildings.geom_type == "MultiPolygon") | (self.Buildings.geom_type == "Polygon")
            ]
        self.Buildings[["x", "y"]] = self.Buildings.centroid.apply(
            lambda b: pd.Series([b.coords[0][0], b.coords[0][1]])
            )
        self.Buildings = pickle.dumps(self.Buildings)
        print(self.city_name, datetime.datetime.now(),'Buildings')


        # Services
        service_columns = ["building_id", "functional_object_id as id", "city_service_type", "center",
                            "city_service_type_id", "city_service_type_code as service_code", "service_name",
                            "address", "capacity", "block_id", "administrative_unit_id", "municipality_id"]
                            
        self.Services = self.get_services(service_columns, add_normative=True, place_slice=place_slice)
        self.Services[["x", "y"]] = self.Services.geometry.apply(
            lambda s: pd.Series([s.coords[0][0], s.coords[0][1]])
            )
        self.PublicTransportStops = self.Services[self.Services["service_code"] == "stops"]
        self.ServiceTypes = pd.read_sql_table(
            "city_service_types", con=self.engine, 
            columns=["id", "code", "public_transport_time_normative", "walking_radius_normative"])

        self.Services = pickle.dumps(self.Services)
        print(self.city_name, datetime.datetime.now(),'Services')
        self.ServiceTypes = pickle.dumps(self.ServiceTypes)
        print(self.city_name, datetime.datetime.now(),'ServiceTypes')
        self.PublicTransportStops = pickle.dumps(self.PublicTransportStops)
        print(self.city_name, datetime.datetime.now(),'PublicTransportStops')


        # Blocks
        blocks_column = ["id", "area", "municipality_id", "administrative_unit_id", "geometry"]
        self.Blocks = self.get_territorial_units("blocks", blocks_column, place_slice=place_slice)
        self.Blocks = pickle.dumps(self.Blocks)
        print(self.city_name, datetime.datetime.now(),'Blocks')


        # Municipalities
        self.Municipalities = self.get_territorial_units("municipalities", ["id", "geometry"], place_slice=place_slice)
        self.Municipalities = pickle.dumps(self.Municipalities)
        print(self.city_name, datetime.datetime.now(),'Municipalities')

        # Districts
        self.AdministrativeUnits = self.get_territorial_units("administrative_units", ["id", "geometry"], place_slice=place_slice)
        self.AdministrativeUnits = pickle.dumps(self.AdministrativeUnits)
        print(self.city_name, datetime.datetime.now(),'AdministrativeUnits')
    
        # Provision
        if self.city_name == "Saint_Petersburg":
            print(self.city_name, datetime.datetime.now(), 'houses_provision')
            houses_provision = pd.read_sql_table("new_houses_provision_tmp", con=self.engine, schema="provision")
            houses_provision = self.transform_provision_file(houses_provision)
            self.houses_provision = houses_provision.rename(
                columns={"administrative_unit_id": "district_id", "municipality_id": "mo_id"})

            print(self.city_name, datetime.datetime.now(),'services_provision')
            services_provision = pd.read_sql("""
                    SELECT p.*, s.administrative_unit_id, s.municipality_id, s.block_id
                    FROM provision.new_services_load_tmp p
                    LEFT JOIN all_services s ON p.functional_object_id=s.functional_object_id""", con=self.engine)
            services_provision = self.transform_provision_file(services_provision)
            self.services_provision = services_provision.rename(
                columns={"administrative_unit_id": "district_id", "municipality_id": "mo_id"})
            chunk_size = 1000
            self.houses_provision = [pickle.dumps(self.houses_provision.iloc[i:i+chunk_size]) for i in range(0, len(self.houses_provision), chunk_size)]
            self.services_provision = [pickle.dumps(self.services_provision.iloc[i:i+chunk_size]) for i in range(0, len(self.services_provision), chunk_size)]
            del houses_provision, services_provision
        else:
            self.houses_provision = pickle.dumps(None)
            self.services_provision = None
        
        print(f"{city_name} is loaded")