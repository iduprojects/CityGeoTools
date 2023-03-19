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

from app.core.config import settings
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
            if settings.UPDATE_CITIES_CACHE or \
                    not self.try_get_city_layers_from_cache_dir(settings.CITIES_CACHE_DIR):
                self.get_city_layers_from_db(settings.CITIES_CACHE_DIR)
            if settings.UPDATE_CITIES_CACHE or \
                    not self.try_get_supplementary_graphs_from_cache(settings.CITIES_CACHE_DIR):
                self.get_supplementary_graphs(settings.CITIES_CACHE_DIR)
        else:
            self.set_none_layers()
        del self.attr_names
        
    def get_city_layers_from_db(self, cities_cache_dir: Optional[str] = None) -> None:

        rpyc_connect = rpyc.connect(
            self.rpyc_adr, self.rpyc_port,
            config={'allow_public_attrs': True, 
                    "allow_pickle": True}
                    )
        rpyc_connect._config['sync_request_timeout'] = None

        cache_failed = False
        for attr_name in self.attr_names:
            print(f"Downloading {self.city_name} - {attr_name} from RPYC")
            binary_data = rpyc_connect.root.get_city_model_attr(self.city_name, attr_name)
            if cities_cache_dir is not None or cache_failed:
                try:
                    if not os.path.isdir(cities_cache_dir):
                        os.makedirs(cities_cache_dir)
                    filename = f"{self.city_name}_{attr_name}.pickle"
                    with open(os.path.join(cities_cache_dir, filename), "wb") as file:
                        file.write(binary_data)
                except Exception as exc:
                    print(f"Could not cache city {self.city_name} data to {filename}: {exc!r}")
                    cache_failed = True
            setattr(self, attr_name, pickle.loads(binary_data))
            
    def try_get_city_layers_from_cache_dir(self, cities_cache_dir: str) -> bool:
        """Try to load city data from the given cache directory.
        
        Return true if the city was successfully loaded, false otherwise."""
        if not os.path.isdir(cities_cache_dir):
            return False
        try:
            for attr_name in self.attr_names:
                if os.path.isfile(filename := os.path.join(cities_cache_dir, f"{self.city_name}_{attr_name}.pickle")):
                    with open(filename, "rb") as file:
                        setattr(self, attr_name, pickle.load(file))
                    print(f"Loaded {self.city_name} - {attr_name} from cities cache directory")
                else:
                    print(f"Missing cache file for city {self.city} - {attr_name}. Redownloading fully.")
                    return False
        except Exception as exc:
            print(f"Got an exception on attempt to read city {self.city_name} from cache dir {cities_cache_dir}: {exc!r}")
            print(f"City {self.city_name} will be redownloaded.")
            return False
        return True
        
    def get_supplementary_graphs(self, cities_cache_dir: Optional[str] = None) -> None:

        sub_edges = ["subway", "bus", "tram", "trolleybus", "walk"] # exclude drive
        MobilitySubGraph = get_subgraph(self.MobilityGraph, "type", sub_edges)
        self.nk_idmap = get_nx2nk_idmap(MobilitySubGraph)
        self.nk_attrs = get_nk_attrs(MobilitySubGraph)
        self.graph_nk_length = convert_nx2nk(MobilitySubGraph, idmap=self.nk_idmap, weight="length_meter")
        self.graph_nk_time = convert_nx2nk(MobilitySubGraph, idmap=self.nk_idmap, weight="time_min")
        self.MobilitySubGraph = load_graph_geometry(MobilitySubGraph)

        if cities_cache_dir is not None:
            try:
                with open(os.path.join(cities_cache_dir, f"{self.city_name}_supplementary_graphs.pickle"), "wb") as file:
                    pickle.dump({
                        "nk_idmap": self.nk_idmap,
                        "nk_attrs": self.nk_attrs,
                        "graph_nk_length": self.graph_nk_length,
                        "graph_nk_time": self.graph_nk_time,
                    }, file)
            except Exception:
                print(f"Could not cache supplementary graphs for city {self.city_name}")

    def try_get_supplementary_graphs_from_cache(self, cities_cache_dir: str) -> None:
        if os.path.isfile(filename := os.path.join(cities_cache_dir, f"{self.city_name}_supplementary_graphs.pickle")):
            try:
                with open(filename, "rb") as file:
                    data = pickle.load(file)
                self.nk_idmap = data["nk_idmap"]
                self.nk_attrs = data["nk_attrs"]
                self.graph_nk_length = data["graph_nk_length"]
                self.graph_nk_time = data["graph_nk_time"]

                sub_edges = ["subway", "bus", "tram", "trolleybus", "walk"] # exclude drive
                self.MobilitySubGraph = load_graph_geometry(get_subgraph(self.MobilityGraph, "type", sub_edges))

                return True
            except Exception as exc:
                print(f"Could not get supplementary graphs for city {self.city_name} from cache: {exc!r}")
                print("Recalculating supplementary graphs.")
                return False
        return False

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