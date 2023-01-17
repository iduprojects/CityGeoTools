import geopandas as gpd
import json

from .utils import nk_routes_between_two_points
from .errors import TerritorialSelectError
from .base_method import BaseMethod


class TrafficCalculator(BaseMethod):

    def __init__(self, city_model):

        BaseMethod.__init__(self, city_model)
        super().validation("traffic_calculator")
        self.stops = self.city_model.PublicTransportStops.copy()
        self.buildings = self.city_model.Buildings.copy()
        self.mobility_graph = self.city_model.graph_nk_length
        self.mobility_graph_attrs = self.city_model.nk_attrs.copy()

    def get_trafic_calculation(self, request_area_geojson):

        living_buildings = self.buildings[self.buildings['population'] > 0]
        living_buildings = living_buildings.loc[:, ('id', 'population', 'geometry')]
        selected_buildings = self._get_custom_polygon_select(request_area_geojson, self.city_crs, living_buildings)[0]

        if len(selected_buildings) == 0:
            raise TerritorialSelectError("living buildings")
        
        stops = self.stops.set_index("id")
        selected_buildings.loc[:, 'nearest_stop_id'] = selected_buildings.apply(
            lambda x: stops.loc[:, 'geometry'].distance(x['geometry']).idxmin(), axis=1)
        nearest_stops = stops.loc[list(selected_buildings.loc[:, 'nearest_stop_id'])]
        path_info = selected_buildings.apply(
            lambda x: nk_routes_between_two_points(self.mobility_graph, self.mobility_graph_attrs,
            p1 = x['geometry'].centroid.coords[0], p2 = stops.loc[x['nearest_stop_id']].geometry.coords[0]), 
            result_type="expand", axis=1)
        house_stop_routes = selected_buildings.copy().drop(["geometry"], axis=1).join(path_info)

        # 30% aprox value of Public transport users
        house_stop_routes.loc[:, 'population'] = (house_stop_routes['population'] * 0.3).round().astype("int")
        house_stop_routes = house_stop_routes.rename(
            columns={'population': 'route_traffic', 'id': 'building_id', "route_geometry": "geometry"})
        house_stop_routes = gpd.GeoDataFrame(house_stop_routes, crs=selected_buildings.crs)

        return {"buildings": json.loads(selected_buildings.reset_index(drop=True).to_crs(4326).to_json()), 
                "stops": json.loads(nearest_stops.reset_index(drop=True).to_crs(4326).to_json()), 
                "routes": json.loads(house_stop_routes.reset_index(drop=True).to_crs(4326).to_json())}