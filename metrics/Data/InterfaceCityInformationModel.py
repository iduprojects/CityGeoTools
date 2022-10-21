from copyreg import pickle
import geopandas as gpd
import networkx as nx
import requests
import pandas as pd
import os
import ast
import numpy as np

from typing import Union
from shapely.geometry import shape
from shapely import wkt
from sqlalchemy import create_engine
from geopandas.geodataframe import GeoDataFrame
from pandas.core.frame import DataFrame
from networkx.classes.multidigraph import MultiDiGraph

import pickle
import rpyc

# TODO: SQL queries as a separate class
# TODO provisions lengths from rpyc method
class InterfaceCityInformationModel:
    
    def __init__(self, city_name, cities_crs, cities_db_id,):

        self.city_name = city_name
        self.city_crs = cities_crs[city_name]
        self.city_id = cities_db_id[city_name]

        self.mongo_address = "http://" + os.environ["MONGO"]
        self.engine = create_engine("postgresql://" + os.environ["POSTGRES"])

        rpyc_server = os.environ["RPYC_SERVER"]
        address, port = rpyc_server.split(":") if ":" in rpyc_server else (rpyc_server, 18861)
        self.rpyc_connect = rpyc.connect(
            address, port,
            config={'allow_public_attrs': True, 
                    "allow_pickle": True}
                    )
        self.rpyc_connect._config['sync_request_timeout'] = None

        self.attr_names = ['walk_graph', 'drive_graph','public_transport_graph',
                           'Buildings', 'Services', 'PublicTransportStops', 
                           'ServiceTypes', 'Blocks', 'Municipalities','AdministrativeUnits']
        self.provisions = ['houses_provision','services_provision', "Social_groups"]
        
        for attr_name in self.attr_names:
            print(attr_name)
            setattr(self, 
                    attr_name, 
                    pickle.loads(self.rpyc_connect.root.get_city_model_attr(city_name, 
                                                                            attr_name)))
        
        for attr_name in self.provisions:
            try:
                if attr_name == 'services_provision':
                    chunk_num_range = 60
                elif attr_name == 'houses_provision':
                    chunk_num_range = 22

                print(self.city_name, attr_name)
                setattr(self, 
                        attr_name, 
                        pd.concat([pickle.loads(self.rpyc_connect.root.get_provisions(city_name,attr_name, chunk_num)) for chunk_num in range(chunk_num_range)]))
            except:
                print(self.city_name, attr_name, "None")
                setattr(self, attr_name, None)

    def get_instagram_data(self, year, season, day_time):
        filename = f'grid_even_num_{year}_top_10_{season}_{day_time}'
        response = requests.get(self.mongo_address + "/uploads/instagram/" + filename)
        file = response.json() if response.status_code == 200 else None
        return file

    def get_service_normative(self, code):
        normative = pd.read_sql(f"""
        SELECT public_transport_time_normative as public_transport,
               walking_radius_normative as walk
               FROM city_service_types 
               WHERE code = '{code}'""", con=self.engine).dropna(axis=1).T.to_records()[0]
        return normative

    def get_living_situation_evaluation(self, situation_id):
        sql_query = f"""
            SELECT AVG(evaluation) evaluation, t.code service_code
            FROM maintenance.living_situations_evaluation e
            JOIN city_service_types t ON t.id = e.city_service_type_id
            WHERE living_situation_id = {situation_id}
            GROUP BY t.code"""
        df = pd.read_sql(sql_query, con=self.engine)
        return df

    def get_service_type(self, id):
        sql_query = f"""
            SELECT t.city_service_type_code
            FROM all_services t
            WHERE t.functional_object_id = '{id}'"""
        df = pd.read_sql(sql_query, con=self.engine)
        return df["city_service_type_code"][0]

    def get_service_code(self, ru_code):
        sql_query = f"""
            SELECT t.code
            FROM city_service_types t
            WHERE t.name = '{ru_code}'"""
        df = pd.read_sql(sql_query, con=self.engine)
        return df["code"][0]
