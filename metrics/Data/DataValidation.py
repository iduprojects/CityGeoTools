import json
import os
import jsonschema

from jsonschema.exceptions import ValidationError
from matplotlib.style import available
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

    def check_methods(self, layer_name, layer):

        print(f"Validation of {layer_name} layer...")
        for method_name in data_dictionary[layer_name]:
            method_class = getattr(self, method_name)
            layer_attr = getattr(method_class, layer_name)
            if "graph" in layer_name:
                layer_attr = method_class.validate_graphml_layers(layer_name, layer)
            else:
                layer_attr = method_class.validate_json_layers(layer_name, layer)

    
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
    
    def validate_graphml_layers(self, layer_name, graph):

        validity = {}
        graph_types = [Graph, DiGraph, MultiGraph, MultiDiGraph]
        validity["networkx_object"] = True if type(graph) in graph_types else False
        validity["not_null_edges"] = True if len(graph.edges()) > 0 else False
        validity["has_geometry"] = all([True if "geometry" in e[-1] else False for e in graph.edges(data=True)])
        validity["has_geometry"] = all([True if "length" in e[-1] else False for e in graph.edges(data=True)])
        validity["has_xy_attr"] = all([True if "x" in n[-1] and "y" in n[-1] else False 
                                            for n in graph.nodes(data=True)])
        setattr(self, layer_name, all(validity.values()))
        bad_features = ", ".join([k for k, v in validity.items() if not v])
        self.message[layer_name] = "Layer matches specification" if all(validity.values()) else f"Error in conditions {bad_features}"
        
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