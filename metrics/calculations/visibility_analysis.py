import geopandas as gpd
import shapely
import json
import shapely.wkt

from .base_method import BaseMethod


class VisibilityAnalysis(BaseMethod):

    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("visibility_analysis")
        self.buildings = self.city_model.Buildings.copy()

    def get_visibility_result(self, point, view_distance):
        
        point_buffer = shapely.geometry.Point(point).buffer(view_distance)
        s = self.buildings.within(point_buffer)
        buildings_in_buffer = self.buildings.loc[s[s].index].reset_index(drop=True)
        buffer_exterior_ = list(point_buffer.exterior.coords)
        line_geometry = [shapely.geometry.LineString([point, ext]) for ext in buffer_exterior_]
        buffer_lines_gdf = gpd.GeoDataFrame(geometry=line_geometry)
        united_buildings = buildings_in_buffer.unary_union

        if united_buildings:
            splited_lines = buffer_lines_gdf['geometry'].apply(lambda x: x.difference(united_buildings))
        else:
            splited_lines = buffer_lines_gdf["geometry"]

        splited_lines_gdf = gpd.GeoDataFrame(geometry=splited_lines).explode(index_parts=True)
        splited_lines_list = []

        for u, v in splited_lines_gdf.groupby(level=0):
            splited_lines_list.append(v.iloc[0]['geometry'].coords[-1])
        circuit = shapely.geometry.Polygon(splited_lines_list)
        if united_buildings:
            circuit = circuit.difference(united_buildings)

        view_zone = gpd.GeoDataFrame(geometry=[circuit]).set_crs(self.city_crs).to_crs(4326)
        return json.loads(view_zone.to_json())