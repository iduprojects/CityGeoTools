import numpy as np

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))

import pandas as pd
import os
import pickle
import rpyc
import json
import geopandas as gpd
import networkx as nx

from sqlalchemy import create_engine
from typing import Optional
from .DataValidation import DataValidation
from .data_transform import load_graph_geometry, convert_nx2nk, get_nx2nk_idmap, get_nk_attrs, get_subgraph

# TODO: SQL queries as a separate class
# TODO provisions lengths from rpyc method
# TODO: there must be one function with common conditions for one graph including walk, personal drive and public
# transport routes.

class CityInformationModel:
    
    def __init__(self, city_name: str, city_crs: int, cities_db_id=Optional[int], cwd="../", mode='user_mode', 
                postgres_con=Optional[str], rpyc_adr=Optional[str], rpyc_port=Optional[int]) -> None:

        self.city_name = city_name
        self.city_crs = city_crs

        self.attr_names = ['MobilityGraph', 'Buildings', 'Services', 
                           'PublicTransportStops', 'ServiceTypes',
                           'RecreationalAreas', 'Blocks', 'Municipalities',
                           'AdministrativeUnits', 'ValueTypes', 'SocialGroups', 
                           'SocialGroupsValueTypesLivingSituations', 
                           'LivingSituationsCityServiceTypes','LivingSituations']
                    
        self.city_id = cities_db_id
        self.cwd = cwd
        self.mode = mode

        if mode == "general_mode":
            self.engine = create_engine(postgres_con)
            self.rpyc_adr = rpyc_adr
            self.rpyc_port = rpyc_port

        self.set_city_layers()
        self.methods = DataValidation() if self.mode == "user_mode" else None

    def __new__(cls, *args, **kwargs):
        if ("mode" not in kwargs or kwargs["mode"] == "user_mode") \
        or (kwargs["mode"] == "general_mode" and cls._validate(*args, **kwargs)):
            return super().__new__(cls)
            

    @classmethod
    def _validate(cls, *args, **kwargs) -> bool:
        rpyc_connect = rpyc.connect(
            kwargs["rpyc_adr"], kwargs["rpyc_port"],
            config={'allow_public_attrs': True, 
                "allow_pickle": True}
                )
        return pickle.loads(rpyc_connect.root.get_city_model_attr(kwargs["city_name"], "readiness"))

    def get_all_attributes(self) -> dict:
        all_attr = self.__dict__
        return all_attr

    def set_city_layers(self) -> None:

        if self.mode == "general_mode":
            self.get_city_layers_from_db()
            self.get_supplementary_graphs()
        else:
            self.set_none_layers()
        del self.attr_names
        
    def get_city_layers_from_db(self) -> None:

        rpyc_connect = rpyc.connect(
            self.rpyc_adr, self.rpyc_port,
            config={'allow_public_attrs': True, 
                    "allow_pickle": True}
                    )
        rpyc_connect._config['sync_request_timeout'] = None
        
        for attr_name in self.attr_names:
            print(self.city_name, attr_name)
            setattr(self, attr_name, pickle.loads(
                rpyc_connect.root.get_city_model_attr(self.city_name, attr_name)))
        
    def get_supplementary_graphs(self) -> None:

        sub_edges = ["subway", "bus", "tram", "trolleybus", "walk"] # exclude drive
        MobilitySubGraph = get_subgraph(self.MobilityGraph, "type", sub_edges)
        self.nk_idmap = get_nx2nk_idmap(MobilitySubGraph)
        self.nk_attrs = get_nk_attrs(MobilitySubGraph)
        self.graph_nk_length = convert_nx2nk(MobilitySubGraph, idmap=self.nk_idmap, weight="length_meter")
        self.graph_nk_time = convert_nx2nk(MobilitySubGraph, idmap=self.nk_idmap, weight="time_min")
        self.MobilitySubGraph = load_graph_geometry(MobilitySubGraph)

    def set_none_layers(self) -> None:
        for attr_name in self.attr_names:
            setattr(self, attr_name, None)

    def update_layers(self, file_dict) -> None:

        for attr_name, file_name in file_dict.items():
            self.update_layer(attr_name, file_name)

    def update_layer(self, attr_name, file_name) -> None:

        if attr_name not in self.get_all_attributes():
            raise ValueError("Invalid attribute name.")

        path, ext = os.path.splitext(file_name)

        if ext == ".graphml":
            graph = nx.read_graphml(file_name, node_type=int)
            graph = load_graph_geometry(graph)
            self.methods.check_methods(attr_name, graph, "validate_graph_layers", self.cwd)
            setattr(self, attr_name, graph)
            self.get_supplementary_graphs()

        elif ext == ".geojson":
            with open(file_name) as f:
                geojson = json.load(f)
            self.methods.check_methods(attr_name,  geojson, "validate_json_layers", self.cwd)
            gdf = gpd.GeoDataFrame.from_features(geojson).set_crs(4326).to_crs(self.city_crs)
            setattr(self, attr_name, gdf)

        elif ext == ".json":
            with open(file_name) as f:
                json_file = json.load(f)
                df = pd.DataFrame(json_file)
            self.methods.check_methods(attr_name,  json_file, "validate_json_layers", self.cwd)
            setattr(self, attr_name, df)
        
        else:
            raise TypeError("Unrecognizable file format.")
        

        print(f"{attr_name} layer loaded successfully!")