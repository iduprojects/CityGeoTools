import networkx as nx
import pandas as pd
import geopandas as gpd
import networkit as nk
import json
from .base_method import BaseMethod

class Blocks_accessibility(BaseMethod):

    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("blocks_accessibility")
        self.blocks = city_model.Blocks.copy()
        self.graph_nk_time =  self.city_model.graph_nk_time
        self.mobility_graph = self.city_model.MobilityGraph.copy()
    
    def get_accessibility(self, target_block: int) -> json:       

        """ 
        The function calculates the accessibility time from specified block to other city blocks.

        :param target_block: specified block from which time to other blocks will be calculated.

        city_model = CityInformationModel(city_name="saint-petersburg", city_crs=32636)
        
        :example: Blocks_accessibility(city_model).get_accessibility(1234)
        
        :return: geojson with block ids and time to them by intermodal graph.
        """

        edges_to_remove = list(((u, v) for u,v,e in self.mobility_graph.edges(data=True) if e['type'] == 'car'))
        self.mobility_graph.remove_edges_from(edges_to_remove)

        self.mobility_graph = nx.convert_node_labels_to_integers(self.mobility_graph)
        graph_df = pd.DataFrame.from_dict(dict(self.mobility_graph.nodes(data=True)), orient='index')
        graph_gdf = gpd.GeoDataFrame(graph_df, geometry =gpd.points_from_xy(graph_df['x'], graph_df['y']),
                                     crs = self.city_crs)

        self.blocks['nearest_node'] = graph_gdf['geometry'].sindex.nearest(self.blocks['geometry'], return_distance = False, 
                                                           return_all = False)[1]
        
        target_node = self.blocks[self.blocks['id'] == target_block]['nearest_node']

        nk_dists = nk.distance.SPSP(G = self.graph_nk_time, sources = [target_node]).run()

        self.blocks['nearest_node'] = self.blocks['nearest_node'].apply(lambda node: 
                                                           round(nk_dists.getDistance(target_node, node), 0))
        
        self.blocks.rename(columns={'nearest_node': 'accessibility_time'}, inplace=True)

        
        return json.loads(self.blocks.to_json())