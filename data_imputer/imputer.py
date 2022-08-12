import warnings
import pandas as pd
import geopandas as gpd
import json
import os
import random
import numpy as np

from typing import Callable
from geopandas.geodataframe import GeoDataFrame
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from datetime import datetime
from numpy.random import random
from tqdm import tqdm, auto

from data_imputer import utils
from data_imputer import prediction
warnings.filterwarnings("ignore")

class DataImputer:

    def __init__(self, data_path: str):

        with open(os.getcwd() + "/data_imputer/config/config_imputation.json") as f:
            self.config_imputation = json.load(f)

        self.projection = self.config_imputation["epsg_projection"]
        self.index_column = self.config_imputation["index_column"] if "index_column" in self.config_imputation else None
        self.file_name, self.file_ext = os.path.splitext(os.path.basename(data_path))
        self.input_data = self.get_data_from_path(data_path)
        self.time_start = str(datetime.now().strftime("%d%m%y_%H%M%S"))

        self.data = self.input_data.copy()
        self.categorical_features = self.config_imputation["categorical_features"]
        self.dtypes = self.data.dtypes.drop("geometry")
        self.nans_position = utils.define_nans_positions(self.data)

        self.imput_counter = 0
        self.iter_counter = 0
        self.num_iter = 0
        self.save_logs = False
        self.save_models = False

        self.imputed_data = None
        self.bunch_scores = None
        self.mean_score = None

    def get_data_from_path(self, data_path: str) -> GeoDataFrame:

        if self.file_ext == ".geojson":
            data = gpd.read_file(data_path).set_crs(self.projection)
            data = data.set_index(self.index_column) if self.index_column else data
        else:
            raise ValueError("Reading from specified file type is not implemented")

        # Check_dtypes function is used to define int attributes and round values subsequently.
        # The function is based on convert_dtypes() from Pandas library.
        # TODO: Find some way to go away from convert_dtypes() because of non-unified data types (like Float64).
        #  These types don't allow to easily save geojson file.
        data = self.check_dtypes(data)
        return data

    @staticmethod
    def check_dtypes(data) -> GeoDataFrame:
        data = data.convert_dtypes()
        dtypes_list = [dt.name for dt in data.dtypes.values.tolist()]
        if "boolean" in dtypes_list:
            data = data.select_dtypes(exclude="boolean").join(data.select_dtypes(include="boolean").astype("Int32"))
        if "string" in dtypes_list:
            data = data.select_dtypes(exclude="string").join(data.select_dtypes(include="string").astype("Int32"))
        return data

    def simulation_omission(self, damage_degree: int, selected_columns: list = None, save: bool = True) -> None:

        selected_columns = self.data.drop(["geometry"], axis=1).columns if not selected_columns else selected_columns
        damaged_data = self.data.apply(
            lambda c: c.mask(random((len(c))) < damage_degree / 100) if c.name in selected_columns else c)
        self.data = damaged_data
        self.nans_position = utils.define_nans_positions(damaged_data)
        save_file_name = "_".join([self.file_name, self.time_start])
        utils.save_to_file(damaged_data.reset_index(), "data_imputer/simulations", save_file_name) if save else None

    def add_neighbors_features(self) -> GeoDataFrame:

        data = self.data.copy()
        if self.config_imputation["search_method"] == "nearest_from_point":
            data = utils.search_neighbors_from_point(data, self.config_imputation["num_neighbors"], 
                                                     self.config_imputation["neighbors_radius"])
        elif self.config_imputation["search_method"] == "touches_boundary":
            data["neighbors"] = data.geometry.apply(lambda x: utils.search_neighbors_from_polygon(x, data.geometry))

        tqdm.pandas(desc="Search for neighbors:")
        data = data.drop(["geometry"], axis=1)
        new_neighbors_features = data.progress_apply(
            lambda x: data.iloc[x["neighbors"]].drop(["neighbors"], axis=1).mean(), axis=1)
        new_neighbors_features = new_neighbors_features.apply(lambda x: x.fillna(x.mean()))
        self.data = self.data.join(new_neighbors_features, rsuffix="_neigh")
        if len(self.categorical_features) > 0:
            self.categorical_features += list(map(lambda x: x + "_neigh", self.categorical_features))


    def multiple_imputation(self, positive_num: bool = True, save_logs: bool = False, save_models: bool = False
                            ) -> GeoDataFrame:

        num_iter = self.config_imputation["num_iteration"]
        num_imput = self.config_imputation["num_imputation"]
        if num_imput < 1 or num_iter < 1:
            raise ValueError("Number imputations and number inner iterations must be equal or more then 1")

        multiple_imputation = []
        data = self.data.copy()
        self.num_iter = num_iter
        self.bunch_scores = {k: [] for k in self.nans_position.keys()}
        num_imputation = len(self.nans_position.keys()) * num_iter

        zero_imputed_data = self.zero_impute(self.config_imputation["initial_imputation_type"])
        sort_data = utils.sort_columns(zero_imputed_data, self.config_imputation["sort_column_type"])
        sorted_semantic_data = sort_data.drop(["geometry"], axis=1)
        base_semantic_columns = list(self.dtypes.index)
        for i in range(1, num_imput + 1):
            self.iter_counter = 0
            self.imput_counter = i
            progress_bar = tqdm(total=num_imputation, position=0, leave=True, desc=f"Iteration of Imputation {i}")
            imputation = self.chained_calculation(
                sorted_semantic_data, progress_bar, positive_num, save_logs, save_models)
            imputation = imputation[base_semantic_columns]
            multiple_imputation.append(imputation)

        imputed_data = pd.concat(multiple_imputation).groupby(level=0).mean() if num_imput > 1 else imputation
        imputed_data = imputed_data.astype(self.dtypes.to_dict())
        imputed_data = imputed_data.join(self.add_flag_columns())
        imputed_data = gpd.GeoDataFrame(imputed_data.join(data.geometry)).set_crs(self.projection)
        utils.save_to_file(imputed_data.reset_index(), "imputed_data", "_".join([self.file_name, self.time_start]))
        self.imputed_data = imputed_data
        self.mean_score = {k: np.mean(v) for k, v in self.bunch_scores.items()}

        return imputed_data

    def zero_impute(self, impute_method: str) -> GeoDataFrame:

        data = self.data
        if impute_method == "distance_weighted":
            data_with_initial_vals = utils.calculate_distance_weighted_values(data, self.nans_position)
        elif impute_method == "mean" or impute_method == "median":
            data_with_initial_vals = utils.calculate_statistics(data, impute_method)
        else:
            raise ValueError("Specified impute type is not implemented.")
        data_with_initial_vals = utils.set_initial_dtypes(self.dtypes, data_with_initial_vals)

        return data_with_initial_vals

    def chained_calculation(self, data: DataFrame, progress_bar: auto.tqdm, positive_num: bool, save_logs: bool,
                            save_models: bool, learn=True, models=None) -> DataFrame:

        if self.iter_counter < self.num_iter:
            self.iter_counter += 1
            self.set_save_options((save_logs, save_models))
            predicted_data = data.apply(lambda x: self.make_iteration(
                x, data, progress_bar, positive_num, learn, models) if x.name in self.nans_position.keys() else x)
            predicted_data = self.chained_calculation(
                predicted_data, progress_bar, positive_num, save_logs, save_models, learn, models)
            return predicted_data
        else:
            return data

    def make_iteration(self, column: Series, data: DataFrame, progress_bar: auto.tqdm, positive_num: bool,
                       learn=True, models=None) -> DataFrame:

        target_name = column.name
        features = data.drop([target_name], axis=1)
        if learn:
            is_categorical = True if target_name in self.categorical_features else False
            args = [is_categorical, positive_num, self.bunch_scores, self.save_logs, self.save_models]
            y_learn = data[target_name].drop(self.nans_position[target_name])
            x_learn = features.loc[y_learn.index]
            x_predict = features.loc[self.nans_position[target_name]]
            y_predict = prediction.learn_and_predict(x_learn, y_learn, x_predict, *args)
        else:
            x_predict = features.loc[self.nans_position[target_name]]
            y_predict = prediction.predict_by_model(models[target_name], x_predict, positive_num)

        column = column.astype("float64")
        column.update(pd.Series(y_predict, index=self.nans_position[column.name]))
        predicted_data = utils.set_initial_dtypes(self.dtypes, column)
        progress_bar.update(1)

        return predicted_data

    def set_save_options(self, save_options: tuple) -> None:
        self.save_logs, self.save_models = save_options if self.num_iter == self.iter_counter else (False, False)
        file_name = f"{self.file_name}_{self.time_start}_{self.imput_counter}"
        if self.save_models:
            os.mkdir(os.path.join(os.getcwd(), "fitted_model", file_name))
        if self.save_logs:
            os.mkdir(os.path.join(os.getcwd(), "logs", file_name))

    def add_flag_columns(self):
        flag_column = self.input_data.drop(["geometry"], axis=1).columns
        flag_data = pd.DataFrame(False, columns=flag_column, index=self.input_data.index)
        flag_data = flag_data.apply(lambda c: c.mask(
            c.index.isin(self.nans_position[c.name]), True
        ) if c.name in self.nans_position.keys() else c)
        flag_data.columns = [c + "_is_imputed" for c in flag_column]
        return flag_data

    def impute_by_saved_models(self, initial_impute_method: str, num_iter: int, positive_num: bool = True) -> GeoDataFrame:

        data = self.data.copy()
        self.num_iter = num_iter
        features_with_models = prediction.parse_config_models(list(data.columns))
        self.nans_position = {k: v for k, v in self.nans_position.items() if features_with_models[k] is not None}
        num_imputation = len(self.nans_position.keys()) * num_iter

        zero_imputed_data = self.zero_impute(initial_impute_method)
        sort_data = utils.sort_columns(zero_imputed_data, "nans_ascending")
        sorted_semantic_data = sort_data.drop(["geometry"], axis=1)
        base_semantic_columns = list(self.dtypes.index)

        progress_bar = tqdm(total=num_imputation, position=0, leave=True, desc=f"Iteration of Imputation")
        imputation = self.chained_calculation(
            sorted_semantic_data, progress_bar, positive_num, save_logs=False, save_models=False, learn=False,
            models=features_with_models)
        imputation = imputation[base_semantic_columns]
        imputed_data = imputation.astype(self.dtypes.to_dict())
        imputed_data = gpd.GeoDataFrame(imputed_data.join(data.geometry)).set_crs(self.projection)
        utils.save_to_file(imputed_data.reset_index(), "imputed_data", "_".join([self.file_name, self.time_start]))

        return imputed_data

    def get_quality_metrics(self, classification_metric: Callable, regression_metric: Callable) -> dict:

        initial_data = self.input_data.copy().drop(["geometry"], axis=1)
        imputed_data = self.imputed_data.copy().drop(["geometry"], axis=1)
        if initial_data.isna().any().any():
            raise NotImplementedError("Initial dataset has omissions. Quality metrics can't be calculated.")

        quality_metrics = {
            k: classification_metric(
                np.array(initial_data[k].loc[v], dtype="float"),
                np.array(imputed_data[k].loc[v], dtype="float")) if k in self.categorical_features else
            regression_metric(np.array(initial_data[k].loc[v], dtype="float"),
                              np.array(imputed_data[k].loc[v], dtype="float")) for k, v in self.nans_position.items()
        }
        path_to_save = os.getcwd() + "/quality_score/quality_score_" + "_".join([self.file_name, self.time_start])
        with open(path_to_save, "w") as file:
            json.dump(quality_metrics, file)
        return quality_metrics