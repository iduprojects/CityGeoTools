import geopandas as gpd
import networkx as nx
import requests
import pandas as pd
import ast

from typing import Union
import shapely
from shapely.geometry import shape
from geopandas.geodataframe import GeoDataFrame
from pandas.core.frame import DataFrame
from networkx.classes.multidigraph import MultiDiGraph

class QueryInterface:

    def get_graph_for_city(self, city: str, graph_type: str, node_type: type) -> MultiDiGraph:

        file_name = city.lower() + "_" + graph_type
        response = requests.get(self.mongo_address + "/uploads/city_graphs/" + file_name)
        if response.status_code == 200:
            graph = nx.readwrite.graphml.parse_graphml(response.text, node_type=node_type)
            return graph
        else:
            return None

    @staticmethod
    def load_graph_geometry(graph: MultiDiGraph) -> MultiDiGraph:

        # for u, v, data in graph.edges(data=True):
        #     data["geometry"] = wkt.loads(data["geometry"])
        
        for u, data in graph.nodes(data=True):
            data["geometry"] = shapely.geometry.Point([data["x"], data["y"]])

        return graph

    def generate_general_sql_query(self, table: str, columns: list, join_tables: str = None, equal_slice: dict = None,
                                   place_slice: dict = None) -> str:
        sql_columns = []
        for c in columns:
            if "center" in c:
                sql_columns.append(c.replace("center", "ST_AsGeoJSON(t.center)"))
            elif "geometry" in c:
                sql_columns.append(c.replace("geometry", "ST_AsGeoJSON(t.geometry) as geometry"))
            else:
                sql_columns.append("t." + c)

        sql_columns = ", ".join(sql_columns)
        sql_columns = sql_columns.replace("t.geometry", "ST_AsGeoJSON(t.geometry)")
        sql_columns = sql_columns.replace("t.center", "ST_AsGeoJSON(t.center)")
        sql_query = f"""SELECT {sql_columns} FROM {table} t """

        sql_query += join_tables if join_tables else ""
        where_statment = ""
        if equal_slice is not None:
            where_statment += f"WHERE {equal_slice['column']} = '{equal_slice['value']}' "
        if place_slice is not None:
            where_statment += "WHERE " if "WHERE" not in where_statment else "and "
            where_statment += self.get_place_slice(place_slice)
        sql_query += where_statment

        return sql_query

    @staticmethod
    def get_place_slice(conditions):

        if conditions["place"] == "polygon":
            slice_row = f"ST_intersects(b.geometry, ST_GeomFromText('POLYGON({polygon}), 4326')) = True"
        elif conditions["place"] == "municipality":
            slice_row = f"municipality_id = {conditions['place_id']}"
        elif conditions["place"] == "district":
            slice_row = f"administrative_unit_id={conditions['place_id']}"
        elif conditions["place"] == "city":
            slice_row = f"city_id = {conditions['place_id']}"
        else:
            raise ValueError("Incorrect area type.")

        return slice_row

    def get_territorial_units(self, territory_type: str, columns: list, place_slice: dict = None
                            ) -> Union[GeoDataFrame, DataFrame]:

        sql_query = self.generate_general_sql_query(territory_type, columns, place_slice=place_slice)
        df = pd.read_sql(sql_query, con=self.engine)
        df = self.del_nan_units(df)

        if "geometry" in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: shape(ast.literal_eval(x)))
            gdf = gpd.GeoDataFrame(df, geometry=df.geometry).set_crs(4326)
            return gdf
        else:
            return df

    def get_buildings(self, columns: list, place_slice: dict = None) -> Union[DataFrame, GeoDataFrame]:

        sql_query = self.generate_general_sql_query("all_buildings", columns, place_slice=place_slice)
        df = pd.read_sql(sql_query, con=self.engine)
        if len(df) > 0:
            df[["x", "y"]] = df["centroid"].apply(lambda x: pd.Series(eval(x)["coordinates"]))

        df = self.del_nan_units(df)
        
        if "geometry" in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: shape(ast.literal_eval(x)))
            gdf = gpd.GeoDataFrame(df, geometry=df.geometry).set_crs(4326)
            gdf = gdf[(gdf.geom_type == "MultiPolygon") | (gdf.geom_type == "Polygon")]
            return gdf
        else:
            return df

    def get_services(self, columns: list, equal_slice: dict = None, 
                    place_slice: dict = None) -> Union[GeoDataFrame, DataFrame]:

        sql_query = self.generate_general_sql_query(
            "all_services", columns, equal_slice=equal_slice, place_slice=place_slice)
        df = pd.read_sql(sql_query, con=self.engine)
        if len(df) > 0:
            df[["x", "y"]] = df["geometry"].apply(lambda x: pd.Series(eval(x)["coordinates"]))

        df = self.del_nan_units(df)

        if "geometry" in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: shape(ast.literal_eval(x)))
            gdf = gpd.GeoDataFrame(df, geometry=df.geometry).set_crs(4326)
            return gdf
        else:
            return df

    # for objects that are out of territorial units for some reason
    @staticmethod
    def del_nan_units(df) -> DataFrame:
        for unit in ["block_id", "municipality_id", "administrative_unit_id"]:
            if unit in df.columns:
                df = df.dropna(subset=[unit])
        return df