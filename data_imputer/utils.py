import pandas as pd
import geopandas as gpd
import numpy as np
import os
import warnings

from typing import Union, Callable
from geopandas.geodataframe import GeoDataFrame
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
tqdm.pandas()


def calculate_distance_weighted_values(data: GeoDataFrame, nans: dict) -> GeoDataFrame:

    coord = data.centroid.apply(lambda point: [point.x, point.y]).tolist()
    geom_column = data["geometry"]
    data = data.drop(["geometry"], axis=1)
    initial_vals = []
    for i in tqdm(range(len(data)), desc="First imputation:"):
        matrix_distance = distance.cdist(coord, np.array([coord[i]]), 'euclidean')
        matrix_distance_div = np.divide(
            1, matrix_distance, out=np.zeros_like(matrix_distance), where=matrix_distance != 0
        )
        distance_weighted_val = np.nansum(
            data.to_numpy(na_value=0) * matrix_distance_div, axis=0
        ) / matrix_distance_div.sum()
        initial_vals.append(distance_weighted_val.tolist())

    initial_vals = pd.DataFrame(initial_vals, columns=data.columns, index=data.index)
    imputed_initial_vals = data.apply(lambda c: c.astype("object").fillna(
        initial_vals[c.name].loc[nans[c.name]].to_dict()) if c.name in nans else c)
    imputed_initial_vals = imputed_initial_vals.join(geom_column).set_geometry("geometry")
    return imputed_initial_vals


def calculate_statistics(data: GeoDataFrame, statistics: str) -> GeoDataFrame:

    geom_column = data["geometry"]
    data = data.drop(["geometry"], axis=1)
    imputed_initial_vals = data.apply(
        lambda c: c.astype("object").fillna(getattr(c, statistics)() if getattr(c, statistics)() is not np.nan else 0))
    imputed_initial_vals = gpd.GeoDataFrame(imputed_initial_vals.join(geom_column), geometry="geometry")
    return imputed_initial_vals


def set_initial_dtypes(dtypes: Series, data_to_round: Union[GeoDataFrame, DataFrame, Series]
                       ) -> Union[GeoDataFrame, DataFrame, Series]:

    dt_neigh = pd.concat([dtypes, dtypes.add_suffix("_neigh")])
    cols = dt_neigh.index
    if type(data_to_round) is Series:
        column_name = data_to_round.name
        rounded_data = data_to_round.round().astype(dt_neigh[column_name].name) \
            if column_name in cols and "Int" in dt_neigh[column_name].name else data_to_round
    else:
        rounded_data = data_to_round.apply(
            lambda c: c.round() if c.name in cols and "Int" in dt_neigh[c.name].name else c)
        rounded_data = rounded_data.apply(lambda c: c.astype(dt_neigh[c.name].name) if c.name in cols else c)
    return rounded_data


def search_neighbors_from_polygon(loc, df):
    bool_series = df.apply(lambda x: loc.touches(x))
    return list(df[bool_series].index)


def search_neighbors_from_point(data, n_neigh, radius):

    x = data.geometry.centroid.apply(lambda point: point.x)
    y = data.geometry.centroid.apply(lambda point: point.y)
    points = pd.DataFrame({"x": x, "y": y})
    neigh = NearestNeighbors(n_neighbors=n_neigh+1, radius=radius)
    neigh.fit(points)
    neigh_dist, neigh_ind = neigh.kneighbors(points)
    neigh_ind = pd.Series(np.delete(neigh_ind, np.s_[:1], 1).tolist())
    neigh_dist = np.delete(neigh_dist, np.s_[:1], 1)
    neigh_dist_mean = pd.Series(neigh_dist.mean(axis=1).tolist())
    data_index = data.index
    data = data.reset_index(drop=True)
    data["dist"] = neigh_dist_mean
    data["neighbors"] = neigh_ind
    return data.set_index(data_index)


def sort_columns(data: Union[DataFrame, GeoDataFrame], sort_type: str) -> Union[DataFrame, GeoDataFrame]:
    if sort_type == "suffering":
        data = data[np.random.choice(data.columns, len(data.columns), replace=False)]
    elif sort_type == "nans_ascending":
        data = data[data.isna().sum().sort_values(ascending=True).index]
    else:
        raise ValueError("Specified sort type is not implemented")
    return data


def define_nans_positions(data: Union[DataFrame, GeoDataFrame]) -> dict:
    imputed_columns = data.columns[data.isna().any()].tolist()
    nans_positions = {k: data[k][data[k].isna()].index for k in imputed_columns}
    return nans_positions


def save_to_file(obj: GeoDataFrame, path: str, filename: str) -> None:

    full_path = os.path.join(os.getcwd(), path, filename + ".geojson")
    obj_to_save = obj.apply(lambda c: pd.Series(
        c.to_numpy(na_value=np.nan, dtype=float), name=c.name, dtype="float") if c.dtype.name != "geometry" else c)
    obj_to_save.to_file(full_path, driver="GeoJSON")

