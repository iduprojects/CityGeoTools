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
from .data_transform import load_graph_geometry, convert_nx2nk, get_nx2_nk_idmap, get_nk_attrs

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
                            'Blocks', 'Municipalities','AdministrativeUnits']
        self.set_city_layers()
        self.methods = DataValidation() if self.mode == "user_mode" else None
    
    def get_all_attributes(self):
        all_attr = self.__dict__
        return all_attr

    def set_city_layers(self):

        if self.mode == "general_mode":
            self.get_city_layers_from_db()
        else:
            self.set_none_layers()
        del self.attr_names
        
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

        self.nk_idmap = get_nx2_nk_idmap(self.MobilityGraph)
        self.nk_attrs = get_nk_attrs(self.MobilityGraph)
        self.graph_nk_length = convert_nx2nk(self.MobilityGraph, idmap=self.nk_idmap, weight="length_meter")
        self.graph_nk_time = convert_nx2nk(self.MobilityGraph, idmap=self.nk_idmap, weight="time_min")
        self.MobilityGraph = load_graph_geometry(self.MobilityGraph)


    def set_none_layers(self):
        for attr_name in self.attr_names:
            setattr(self, attr_name, None)

    def update_layers(self, file_dict):

        for attr_name, file_name in file_dict.items():
            self.update_layer(attr_name, file_name)

    def update_layer(self, attr_name, file_name):

        if attr_name not in self.get_all_attributes():
            raise ValueError("Invalid attribute name.")

        if  attr_name == "MobilityGraph":
            graph = nx.read_graphml(file_name, node_type=int)
            graph = load_graph_geometry(graph)
            self.methods.check_methods(attr_name, graph, "validate_graph_layers")
            setattr(self, attr_name, graph)
            self.nk_idmap = get_nx2_nk_idmap(graph)
            self.nk_attrs = get_nk_attrs(graph)
            self.graph_nk_length = convert_nx2nk(graph, weight="length_meter")
            self.graph_nk_time = convert_nx2nk(graph, weight="time_min")

        else: 
            with open(file_name) as f:
                geojson = json.load(f)
            self.methods.check_methods(attr_name,  geojson, "validate_json_layers")
            gdf = gpd.GeoDataFrame.from_features(geojson).set_crs(4326).to_crs(self.city_crs)
            setattr(self, attr_name, gdf)
        

        print(f"{attr_name} layer loaded successfully!")