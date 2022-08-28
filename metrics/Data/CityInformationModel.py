from ctypes import util
import pandas as pd
import os
import pickle
import rpyc
import json
import geopandas as gpd
import networkx as nx

from sqlalchemy import create_engine
from .DataValidation import DataValidation
from ..Calculations import utils

# TODO: SQL queries as a separate class
# TODO provisions lengths from rpyc method
# TODO: there must be one function with common conditions for one graph including walk, personal drive and public
# transport routes.

class CityInformationModel:
    
    def __init__(self, city_name, city_crs, cities_db_id=None, mode='user_mode'):

        self.city_name = city_name
        self.city_crs = city_crs
        self.city_id = cities_db_id
        self.mode = mode

        self.attr_names = ['MobilityGraph', 'Buildings', 'Services', 'PublicTransportStops',
                            'Blocks', 'Municipalities','Administrative_units']
        self.provisions = ['houses_provision','services_provision']
        self.set_city_layers()
        self.methods = DataValidation() if self.mode == "user_mode" else None
    
    def get_all_attributes(self):
        all_attr = self.__dict__
        del all_attr["attr_names"], all_attr["provisions"]
        return all_attr

    def set_city_layers(self):

        if self.mode == "general_mode":
            self.get_city_layers_from_db()
        else:
            self.set_none_layers()
        
    def get_city_layers_from_db(self):

        self.engine = create_engine("postgresql://" + os.environ["POSTGRES"])
        rpyc_connect = rpyc.connect(
            os.environ["RPYC_SERVER"], 18861,
            config={'allow_public_attrs': True, 
                    "allow_pickle": True}
                    )
        rpyc_connect._config['sync_request_timeout'] = None
        
        for attr_name in self.attr_names:
            print(self.city_name, attr_name)
            setattr(self, attr_name, pickle.loads(
                rpyc_connect.root.get_city_model_attr(self.city_name, attr_name)))

            self.graph_nk_length = utils.convert_nx2nk(self.MobilityGraph, weight="length_meter")
            self.graph_nk_time = utils.convert_nx2nk(self.MobilityGraph, weight="time_min")
        
        for attr_name in self.provisions:
            try:
                if attr_name == 'services_provision':
                    chunk_num_range = 60
                elif attr_name == 'houses_provision':
                    chunk_num_range = 22

                print(self.city_name, attr_name)
                setattr(self, 
                        attr_name, 
                        pd.concat([pickle.loads(
                            self.rpyc_connect.root.get_provisions(self.city_name, attr_name, chunk_num)) 
                            for chunk_num in range(chunk_num_range)]))
            except:
                print(self.city_name, attr_name, "None")
                setattr(self, attr_name, None)

    def set_none_layers(self):
        for attr_name in self.attr_names + self.provisions:
            setattr(self, attr_name, None)

    def update_layers(self, file_dict):

        for attr_name, file_name in file_dict.items():
            self.update_layer(attr_name, file_name)

    def update_layer(self, attr_name, file_name):

        if attr_name not in self.get_all_attributes():
            raise ValueError("Invalid attribute name.")

        if  attr_name == "mobility_graph":
            graph = nx.read_graphml(file_name, node_type=int)
            self.methods.check_methods(attr_name, graph, "validate_graph_layers")
            setattr(self, attr_name, graph)
            self.graph_nk_length = utils.convert_nx2nk(graph, weight="length_meter")
            self.graph_nk_time = utils.convert_nx2nk(graph, weight="time_min")

        else: 
            with open(file_name) as f:
                geojson = json.load(f)
            self.methods.check_methods(attr_name,  geojson, "validate_json_layers")
            gdf = gpd.GeoDataFrame.from_features(geojson).set_crs(4326).to_crs(self.city_crs)
            setattr(self, attr_name, gdf)
        

        print(f"{attr_name} layer loaded successfully!")

# ####################################################################################################

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

    def get_social_groups_significance(self, user_request):
        social_groups_significance = requests.get(self.db_api_provision +
            "/api/relevance/service_types/?social_group={social_group_name}".format(
                    social_group_name=user_request['user_social_group_selection'])
                    ).json()['_embedded']['service_types']
        return social_groups_significance
