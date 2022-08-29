import geopandas as gpd
import networkx as nx
import requests
import pandas as pd
import ast
import numpy as np

from typing import Union
import shapely
from shapely.geometry import shape
from shapely import wkt
from geopandas.geodataframe import GeoDataFrame
from pandas.core.frame import DataFrame
from networkx.classes.multidigraph import MultiDiGraph


class QueryInterface:

    def get_graph_for_city(self, city: str, graph_type: str, node_type: type) -> MultiDiGraph:

        file_name = city.lower() + "_" + graph_type
        graph = requests.get(self.mongo_address + "/uploads/city_graphs/" + file_name)
        graph = nx.readwrite.graphml.parse_graphml(graph.text, node_type=node_type)
        if graph_type == "walk_graph" or graph_type == "drive_graph":
            graph = self.load_graph_geometry(graph)
        return graph

    @staticmethod
    def load_graph_geometry(graph: MultiDiGraph) -> MultiDiGraph:

        # for u, v, data in graph.edges(data=True):
        #     data["geometry"] = wkt.loads(data["geometry"])
        
        for u, data in graph.nodes(data=True):
            data["geometry"] = shapely.geometry.Point([data["x"], data["y"]])

        return graph


    def generate_general_sql_query(self, table: str, columns: list, join_tables: str = None, equal_slice: dict = None,
                                   place_slice: dict = None) -> str:

        columns = ", ".join(columns)
        columns = columns.replace("t.geometry", "ST_AsGeoJSON(t.geometry) AS geometry")
        columns = columns.replace("t.center", "ST_AsGeoJSON(t.center) AS geometry")
        sql_query = f"""SELECT {columns} FROM {table} t """

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

        columns = ["t." + c for c in columns]
        sql_query = self.generate_general_sql_query(territory_type, columns, place_slice=place_slice)
        df = pd.read_sql(sql_query, con=self.engine)
        if "geometry" in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: shape(ast.literal_eval(x)))
            gdf = gpd.GeoDataFrame(df, geometry=df.geometry).set_crs(4326).to_crs(self.city_crs)
            return gdf
        else:
            return df

    def get_buildings(self, columns: list, place_slice: dict = None) -> Union[DataFrame, GeoDataFrame]:

        columns = ["t." + c for c in columns]
        sql_query = self.generate_general_sql_query("all_buildings", columns, place_slice=place_slice)
        df = pd.read_sql(sql_query, con=self.engine)
        if "geometry" in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: shape(ast.literal_eval(x)))
            gdf = gpd.GeoDataFrame(df, geometry=df.geometry).set_crs(4326).to_crs(self.city_crs)
            return gdf
        else:
            return df

    def get_services(self, columns: list, equal_slice: dict = None, place_slice: dict = None,
                     add_normative: bool = False) -> Union[GeoDataFrame, DataFrame]:

        join_table = ""
        columns = ["t." + c for c in columns]
        if add_normative:
            columns.extend(['s.houses_in_radius', 's.people_in_radius', 's.service_load',
                            's.needed_capacity AS loaded_capacity', 's.reserve_resource', 'n.normative'])
            join_table = f"""
            LEFT JOIN provision.services s ON functional_object_id = s.service_id 
            LEFT JOIN provision.normatives n ON functional_object_id = n.normative """

        sql_query = self.generate_general_sql_query(
            "all_services", columns, join_tables=join_table, equal_slice=equal_slice, place_slice=place_slice)
        sql_query = sql_query.replace("functional_object_id,", "functional_object_id AS index,")
        df = pd.read_sql(sql_query, con=self.engine)
        if "geometry" in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: shape(ast.literal_eval(x)))
            gdf = gpd.GeoDataFrame(df, geometry=df.geometry).set_crs(4326).to_crs(self.city_crs)
            return gdf
        else:
            return df

    def transform_provision_file(self, table):
        table = gpd.GeoDataFrame(table, geometry = table['geometry'].apply(lambda x: shapely.wkt.loads(x)), crs=4326).to_crs(self.city_crs)
        table = table.apply(lambda col: col.apply(lambda row: self.get_eval_values(row)))
        table = table.set_index("functional_object_id").replace("NaN", np.nan)
        return table

    @staticmethod
    def get_eval_values(value):
        try:
            parsed_value = eval(value)
            if type(parsed_value) in [str, int, float, dict, list]:
                return parsed_value
            else:
                return value
        except:
            return value
