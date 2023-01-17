import pandas as pd
import json
import numpy as np
import networkit as nk

from scipy import spatial
from .errors import SelectedValueError, TerritorialSelectError
from .base_method import BaseMethod
from .mobility_analysis import AccessibilityIsochrones


class Diversity(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("diversity")
        self.mobility_graph_length = self.city_model.graph_nk_length
        self.mobility_graph_time = self.city_model.graph_nk_time
        self.graph_attrs = self.city_model.nk_attrs.copy()
        self.services = self.city_model.Services.copy()
        self.service_types = self.city_model.ServiceTypes.copy()
        self.municipalities = self.city_model.Municipalities.copy()
        self.blocks = self.city_model.Blocks.copy()

        self.buildings = self.city_model.Buildings.copy()
        self.living_buildings = self.buildings[self.buildings['is_living'] == True].reset_index(drop=True)
        if len(self.living_buildings) == 0:
            raise TerritorialSelectError("living buildings")

    def get_diversity(self, service_type, geojson):
        services_select = self.services[self.services["service_code"] == service_type]
        if len(services_select) == 0: raise SelectedValueError("services", service_type, "service_code") 

        houses = self.living_buildings
        if geojson: houses = self._get_custom_polygon_select(geojson, self.city_crs, houses)[0]
        if len(houses) == 0: raise TerritorialSelectError("houses") 

        travel_type, weigth, limit_value, graph = self._define_service_normative(service_type)
        dist_matrix = self._get_distance_matrix(houses, services_select, graph, limit_value)
        houses = self._calculate_diversity(houses, dist_matrix)

        blocks = self.blocks.dropna(subset=["municipality_id"]) # TEMPORARY
        blocks = self.blocks.join(houses.groupby(["block_id"])["diversity"].mean().round(2), on="id", how="inner")
        municipalities = self.municipalities.join(
            houses.groupby(["municipality_id"])["diversity"].mean().round(2), on="id", how="inner"
            )
        return {
            "municipalities": json.loads(municipalities.to_crs(4326).fillna("None").to_json()),
            "blocks": json.loads(blocks.to_crs(4326).fillna("None").to_json())
                    }

    def get_houses(self, block_id, service_type):

        services = self.services[self.services["service_code"] == service_type]
        if len(services) == 0:
            raise SelectedValueError("services", service_type, "service_code")

        houses_in_block = self.living_buildings[self.living_buildings['block_id'] == block_id].reset_index(drop=True)
        if len(houses_in_block) == 0:
            raise TerritorialSelectError("living buildings")

        travel_type, weigth, limit_value, graph = self._define_service_normative(service_type)
        dist_matrix = self._get_distance_matrix(houses_in_block, services, graph, limit_value)
        houses_in_block = self._calculate_diversity(houses_in_block, dist_matrix)

        return json.loads(houses_in_block.to_crs(4326).to_json())

    def get_info(self, house_id, service_type):

        services = self.services[self.services["service_code"] == service_type]
        if len(services) == 0:
            raise SelectedValueError("services", service_type, "service_code")

        house = self.living_buildings[self.living_buildings['id'] == house_id].reset_index(drop=True)
        if len(house) == 0:
            raise SelectedValueError("living building", house_id, "id")
            
        house_x, house_y = house[["x", "y"]].values[0]
        travel_type, weigth, limit_value, graph = self._define_service_normative(service_type)
        dist_matrix = self._get_distance_matrix(house, services, graph, limit_value)
        house = self._calculate_diversity(house, np.vstack(dist_matrix[0]))
        selected_services = services[dist_matrix[:, 0] == 1]
        isochrone = AccessibilityIsochrones(self.city_model).get_accessibility_isochrone(
            travel_type, house_x, house_y, limit_value, weigth)
        return {
            "house": json.loads(house.to_crs(4326).to_json()),
            "services": json.loads(selected_services.to_crs(4326).to_json()),
            "isochrone": isochrone["isochrone"]
        }

    def _define_service_normative(self, service_type):

        service_type_info = self.service_types[self.service_types["code"] == service_type]

        if service_type_info["walking_radius_normative"].notna().values:
            travel_type = "walk"
            weigth = "length_meter"
            limit_value = service_type_info["walking_radius_normative"].values[0]
            graph = self.mobility_graph_length
        elif service_type_info["public_transport_time_normative"].notna().values:
            travel_type = "public_transport"
            weigth = "time_min"
            limit_value = service_type_info["public_transport_time_normative"].values[0]
            graph = self.mobility_graph_time
        else:
            raise ValueError("Any service type normative is None.")

        return travel_type, weigth, limit_value, graph
        
    def _get_distance_matrix(self, houses, services, graph, limit_value):

        houses_distance, houses_nodes = spatial.cKDTree(self.graph_attrs).query([houses[["x", "y"]]])
        services_distance, services_nodes = spatial.cKDTree(self.graph_attrs).query([services[["x", "y"]]])

        if len(services_nodes[0]) < len(houses_nodes[0]):
            source, target = services_nodes, houses_nodes
            source_dist, target_dist = services_distance, houses_distance
        else:
            source, target = houses_nodes, services_nodes
            source_dist, target_dist = houses_distance, services_distance

        dijkstra = nk.distance.SPSP(graph, source[0])
        dijkstra = dijkstra.run()
        dist_matrix = dijkstra.getDistances(asarray=True)
        dist_matrix = dist_matrix[:, target[0]] + target_dist[0] + np.vstack(np.array(source_dist[0]))
        dist_matrix = np.where(dist_matrix > limit_value, dist_matrix, 1)
        dist_matrix = np.where(dist_matrix <= limit_value, dist_matrix, 0)
    
        return dist_matrix if len(services_nodes[0]) < len(houses_nodes[0]) else dist_matrix.T

    @staticmethod
    def _calculate_diversity(houses, dist_matrix):
        
        count_services = dist_matrix.sum(axis=0)
        diversity_estim = {1:0.2, 2:0.4, 3:0.6, 4:0.8, 5:1}
        for count, estim in diversity_estim.items():
            count_services[count_services == count] = estim
        count_services = np.where(count_services < 5, count_services, 1)
        houses_diversity = pd.Series(count_services, index=houses["id"]).rename("diversity")
        houses = houses.join(houses_diversity, on="id")
        return houses