import json
import importlib
import os
import jsonschema
import geopandas as gpd

from jsonschema.exceptions import ValidationError
from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph
from networkx.classes.digraph import DiGraph
from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.multigraph import MultiGraph

from .data_dictionary import data_dictionary


class DataValidation:
    
    def __init__(self):

        self.traffic_calculator = TrafficCalculatorData()
        self.visibility_analysis = VisibilityAnalysisData()
        self.weighted_voronoi = WeightedVoronoiData()
        self.blocks_clusterization = BlocksClusterizationData()
        self.services_clusterization = ServicesClusterizationData()
        self.spacematrix = SpacematrixData()
        self.accessibility_isochrones = AccessibilityIsochronesData()

    def check_methods(self, layer_name, validate_object, validation_func):

        print(f"Validation of {layer_name} layer...")
        for method_name in data_dictionary[layer_name]:
            method_class = getattr(self, method_name)
            validation_func = getattr(method_class, validation_func)
            validation_func(layer_name, validate_object)
    
    def validate_json_layers(self, layer_name, layer):

        file = layer_name + ".json"
        with open(os.path.join(self.specification_folder, file)) as schema:
            schema = json.load(schema)

        try:
            jsonschema.validate(instance=layer, schema=schema)
            setattr(self, layer_name, True)
            self.message[layer_name] = "Layer matches specification"

        except ValidationError as error:
            setattr(self, layer_name, False)
            self.message[layer_name] = error.message

        return gpd.GeoDataFrame.from_features(layer).set_crs(4326).to_crs(32636)
    
    def validate_graph_layers(self, layer_name, graph):

        path = self.specification_folder.replace("/", ".")
        mod = importlib.import_module(".mobility_graph", path)
        node_validity, edge_validity = mod.validate_graph(graph)
        validity = all(node_validity.values()) & all(edge_validity.values())
        setattr(self, layer_name, validity)

        edge_error = ", ".join([k for k, v in edge_validity.items() if not v])
        node_error = ", ".join([k for k, v in node_validity.items() if not v])

        self.message[layer_name] = "Layer matches specification" if validity else ""
        self.message[layer_name] += f"Edges do not have {edge_error} attributes. " if len(edge_error) > 0 else ""
        self.message[layer_name] += f"Nodes do not have {node_error} attributes." if len(node_error) > 0 else ""
        
    def get_list_of_methods(self):
        return list(self.__dict__.keys())

    def if_method_available(self, method):
        method_data = getattr(self, method).__dict__.items()
        return all([v for k, v in method_data if k not in ["specification_folder", "messag"]])

    def get_bad_layers(self, method):
        method_data = getattr(self, method).__dict__.items()
        return [k for k, v in method_data if k not in ["specification_folder", "message"] and not v]

    def get_list_of_available_methods(self):
        return [method for method in list(self.__dict__.keys()) if self.if_method_available(method)]


class TrafficCalculatorData(DataValidation):
    def __init__(self):
        self.specification_folder = "data_specification/traffic_calculator"
        self.Buildings = None
        self.Public_Transport_Stops = None
        self.walk_graph = None
        self.message = {}


class VisibilityAnalysisData(DataValidation):
    def __init__(self):
        self.specification_folder = "data_specification/visibility_analysis"
        self.Buildings = None  
        self.message = {}

class WeightedVoronoiData(DataValidation):
    def __init__(self):
        self.specification_folder = None
        self.message = "No data are nedded"

class BlocksClusterizationData(DataValidation):
    def __init__(self):
        self.specification_folder = "data_specification/blocks_clusterization"
        self.Services = None
        self.Blocks = None
        self.message = {}

class ServicesClusterizationData(DataValidation):
    def __init__(self):
        self.specification_folder = "data_specification/services_clusterization"
        self.Services = None
        self.message = {}

class SpacematrixData(DataValidation):
    def __init__(self):
        self.specification_folder = "data_specification/spacematrix"
        self.Buildings = None
        self.Blocks = None
        self.message = {}
    
class AccessibilityIsochronesData(DataValidation):
    def __init__(self):
        self.specification_folder = "data_specification/accessibility_isochrones"
        self.Buildings = None
        self.Blocks = None
        self.message = {}