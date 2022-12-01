import pandas as pd
import numpy as np
import os
import glob
import json
import sklearn

from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.metrics import make_scorer
from joblib import dump, load
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from os.path import join
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.exceptions import NotFittedError

from data_imputer import utils

"""If column name is in categorical_features list classification predictive method will be used.
For other columns will be used regression predictive method.
To find the best combination hyperparametrs we use sklearn implementation of Successive Halving algorithm. 
Candidate parameter values are specified by user in config/config_learning.json."""


def learn_and_predict(x_learn: DataFrame, y_learn: Series, x_predict: DataFrame, is_categorical: bool, is_positive: bool,
                      bunch_scores: list, save_log: bool, save_model: bool):

    model = build_model(x_learn, y_learn, is_categorical, bunch_scores, save_model, save_log)
    prediction = predict_by_model(model, x_predict, is_positive)
    return prediction


def build_model(x_learn: DataFrame, y_learn: Series, is_categorical: bool, scores: list,
                save_model: bool, save_log: bool):

    path = os.getcwd()
    with open(path + "/CityGeoTools/data_imputer/config/config_learning.json") as f:
        config = json.load(f)

    method = "classification" if is_categorical else "regression"
    pipe = Pipeline([("transform", eval(config[method]["transform"])), ('model', eval(config[method]["model"]))])
    gbdt_search = config[method]["grid_search_param"]
    gbdt_search["model"] = eval(gbdt_search["model"])
    gbdt_search["model__max_features"] = eval(gbdt_search["model__max_features"])
    learn_param = config[method]["learn_param"]
    learn_param["scoring"]["score_func"] = eval(learn_param["scoring"]["score_func"])
    scorer = make_scorer(**learn_param["scoring"], greater_is_better=False)
    learn_param["scoring"] = scorer
    opt = HalvingGridSearchCV(pipe, gbdt_search, **learn_param)
    try:
        opt.fit(x_learn.to_numpy(), y_learn.astype("float").to_numpy())
        scores[y_learn.name].append(opt.best_score_)
        save_model_object(opt, y_learn.name) if save_model else None
        save_logs_object(opt, y_learn.name) if save_log else None
    except NotFittedError:
        print("NotFittedError was raised")
        opt = build_model(x_learn, y_learn, is_categorical, scores, save_model, save_log)

    return opt


def predict_by_model(model, x_predict: DataFrame, is_positive: bool):
    y_predict = model.predict(x_predict)
    y_predict = [0 if y < 0 else y for y in y_predict] if is_positive else y_predict
    return y_predict


def parse_config_models(columns: list) -> dict:
    path = os.getcwd()
    with open(path + "/CityGeoTools/data_imputer/config/config_prediction.json") as f:
        config = json.load(f)

    folder = os.path.join(path, config["folder"])
    existed_models = os.listdir(folder)
    existed_models = {k: next((m for m in existed_models if k in m), None) for k in columns}
    existed_model_objects = {k: load(os.path.join(folder, v)) if v is not None else v for k, v in existed_models.items()}
    return existed_model_objects


def save_logs_object(model, model_name: str) -> None:
    cv_result = pd.DataFrame(model.cv_results_)
    last_created_folder = max(glob.glob(os.path.join(os.getcwd(), "/CityGeoTools/data_imputer/logs", "*/")), key=os.path.getmtime)
    log_file = os.path.join(last_created_folder, model_name + ".xlsx")
    cv_result.to_excel(log_file, engine="openpyxl")


def save_model_object(model, model_name: str) -> None:
    last_created_folder = max(glob.glob(os.path.join(os.getcwd(), "/CityGeoTools/data_imputer/fitted_model", "*/")), 
                                key=os.path.getmtime)
    path = os.path.join(last_created_folder, model_name + "_GBDT.joblib")
    dump(model, path)



