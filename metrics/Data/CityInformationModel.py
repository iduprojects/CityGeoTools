import json
from attr import attr
from fastapi import File
from matplotlib.style import available
import pandas as pd
import os
import pickle
import rpyc
import jsonschema
import geopandas as gpd

from sqlalchemy import create_engine
from jsonschema.exceptions import ValidationError
from copyreg import pickle

# TODO: SQL queries as a separate class
# TODO provisions lengths from rpyc method

class CityInformationModel:
    
    def __init__(self, city_name, city_crs, cities_db_id=None, mode='user_mode'):

        self.city_name = city_name
        self.city_crs = city_crs
        self.city_id = cities_db_id
        self.mode = mode

        self.attr_names = ['walk_graph', 'drive_graph','public_transport_graph',
                            'Buildings','Spacematrix_Buildings', 'Services',
                            'Public_Transport_Stops','Spacematrix_Blocks',
                            'Block_Diversity', 'Base_Layer_Blocks',
                            'Base_Layer_Municipalities','Base_Layer_Districts']
        self.provisions = ['houses_provision','services_provision']
        self.set_city_layers()
        self.available_mathods = self.MethodFlags()
    
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
                self.rpyc_connect.root.get_city_model_attr(self.city_name, attr_name)))
        
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

    def update_layer(self, attr_name, layer):
        self.available_mathods.validate_json_layers(attr_name, layer)
        py_obj = self.get_pandas_object(attr_name, layer)
        setattr(self, attr_name, py_obj)
        print(f"{attr_name} layer loaded successfully!")

    def get_pandas_object(self, attr_name, layer):

        if "type" in layer.keys() and layer["type"] == 'FeatureCollection':
            print(f"Converting {attr_name} layer to GeoDataFrame...")
            return gpd.GeoDataFrame.from_features(layer)

    class MethodFlags:
        
        def __init__(self):
            
            self.specification_folder = "data_specification"

            self.traffic_calculator = {"Buildings": None, "Public_Transport_Stops": None}
            self.visibility_analysis = {}
            self.weighted_voronoi = {}
            self.spacematrix = {}

            self.diversity = {}
            self.provision = {}
            self.wellbeing = {}

            self.walk_drive_isochrone = {}
            self.public_transport_isochrone = {}
        
        def validate_json_layers(self, file_name, layer):

            print(f"Validation of {file_name} layer...")
            for dir in os.listdir(self.specification_folder):
                file = file_name + ".json"
                
                if file in os.listdir(os.path.join(self.specification_folder, dir)):
                    with open(os.path.join(self.specification_folder, dir, file)) as schema:
                        schema = json.load(schema)
                    try:
                        jsonschema.validate(instance=layer, schema=schema)
                        method = getattr(self, dir)
                        method[file_name] = True
                    except ValidationError:
                        method = getattr(self, dir)
                        method[file_name] = False
        
        def get_list_of_methods(self):
            attrs = self.__dict__.copy()
            del attrs["specification_folder"]
            return list(attrs.keys())

        def get_list_of_available_methods(self):
            attrs = self.__dict__.copy()
            del attrs["specification_folder"]
            return [method for method, files in attrs.items() if all(files.values()) and len(files) > 0]
        
        def if_method_available(self, method):
            return all(getattr(self, method).values())

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
