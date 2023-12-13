import jsonschema
import json
import requests
import logging

from .logger import logger


class DataValidation():

    def __init__(self, city_name, mongo_address):

        self.city_name = city_name
        self.mongo_address = mongo_address

        self.MobilityGraph = True
        self.Buildings = True
        self.ServiceTypes = True
        self.Services = True
        self.PublicTransportStops = True
        self.Blocks = True
        self.Municipalities = True
        self.AdministrativeUnits = True


    def validate_df(self, layer_name, df, file_type):

        if file_type == "geojson":
            json_obj = json.loads(df.to_json())
            json_obj["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}} 
        else: 
            json_obj = json.loads(df.to_json(orient="records"))
            
        schema = requests.get(self.mongo_address + f"/uploads/specification/{layer_name}", timeout=600).json()
        validator = jsonschema.Draft7Validator(schema)
        if validator.is_valid(json_obj):
            setattr(self, layer_name, True)
            logger.info(f"{self.city_name} - {layer_name} matches specification.")
        else:
            setattr(self, layer_name, False)
            errors = validator.iter_errors(json_obj)
            messages = set([
                f"{e.relative_path[-1]}: {e.message}" if e.relative_path[-1] != "features"
                else "Some property in 'features' contain only NaN. Or there is no objects in 'features' at all." 
                for e in errors])
            for message in messages:
                    logger.critical(f"{self.city_name} - {layer_name} DO NOT match specification. {message}.")
            

    def validate_graphml(self, layer_name, G):
        
        message = []
        edge_validity = {}
        node_validity = {}
        public_transport = ["subway", "tram", "trolleybus", "bus"]

        if not G:
            setattr(self, layer_name, False)
            logger.critical(f"{self.city_name} - Intermodal graph doesn't exist.")
            return None

        graph_size = len(G.edges()) > 1
        types = set([e[-1]["type"] for e in G.edges(data=True)])
        edge_validity["type"] = len(types) > 0
        edge_validity["walk value in type"] = 'walk' in types
        edge_validity["public transport in type"] = any([t in types for t in public_transport])
        edge_validity["length_meter"] = all(["length_meter" in e[-1] for e in G.edges(data=True)])
        edge_validity["time_min"] = all(["time_min" in e[-1] for e in G.edges(data=True)])

        node_validity["x"] = all(["x" in n[-1] for n in G.nodes(data=True)])
        node_validity["y"] = all(["y" in n[-1] for n in G.nodes(data=True)])
        node_validity["stop"] = all(["stop" in n[-1] for n in G.nodes(data=True)])

        validity = graph_size & all(node_validity.values()) & all(edge_validity.values())
        if validity:
            setattr(self, layer_name, validity)
            logging.info(f"{self.city_name} - {layer_name} matches specification.")
        else: 
            edge_error = ", ".join([k for k, v in edge_validity.items() if not v])
            node_error = ", ".join([k for k, v in node_validity.items() if not v])
            message = "Layer matches specification" if validity else ""
            message += f"Graph has too little edges." if not graph_size else ""
            message += f"Edges do not have {edge_error} attributes. " if len(edge_error) > 0 else ""
            message += f"Nodes do not have {node_error} attributes." if len(node_error) > 0 else ""

            setattr(self, layer_name, False)
            logger.critical(f"{self.city_name} - {layer_name} DO NOT match specification. {message}")