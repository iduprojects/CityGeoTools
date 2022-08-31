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
        self.MobilityGraph = self.get_graph_for_city(city_name, "intermodal_graph", node_type=int)
        self.MobilityGraph = pickle.dumps(self.MobilityGraph)
        print(self.city_name, datetime.datetime.now(),'intermodal_graph')


        # Buildings
        buildings_columns = ["building_id as id", "building_area as basement_area", "is_living", "living_area", 
                            "population_balanced as population", "storeys_count", "administrative_unit_id", 
                            "municipality_id", "block_id", "geometry"]

        self.Buildings = self.get_buildings(buildings_columns, place_slice).to_crs(self.city_crs)
        self.Buildings = self.Buildings[
            (self.Buildings.geom_type == "MultiPolygon") | (self.Buildings.geom_type == "Polygon")
            ]
        self.Buildings = pickle.dumps(self.Buildings)
        print(self.city_name, datetime.datetime.now(),'Buildings')


        # Services
        service_columns = ["building_id", "functional_object_id as id", "city_service_type", "center",
                            "city_service_type_id", "city_service_type_code as service_code", "service_name",
                            "block_id", "administrative_unit_id", "municipality_id"]
                            
        self.Services = self.get_services(service_columns, place_slice=place_slice)
        self.PublicTransportStops = self.Services[self.Services["service_code"] == "stops"]
        self.Services = pickle.dumps(self.Services)
        self.PublicTransportStops = pickle.dumps(self.PublicTransportStops)
        print(self.city_name, datetime.datetime.now(),'Services')
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