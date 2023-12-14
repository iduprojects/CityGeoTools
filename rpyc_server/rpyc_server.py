import rpyc
import os
import datetime
import pandas as pd
import pickle

from data_classes.logger import logger
from sqlalchemy import create_engine
from rpyc.utils.server import ThreadedServer
from data_classes.InterfaceCityInformationModel import DataQueryInterface


class MyService(rpyc.Service):
    ready_cities = []

    def get_city_model_attr(self, city_name, atr_name):
        print(city_name, datetime.datetime.now(), atr_name)
        return getattr(city_models[city_name], atr_name)


if __name__ == "__main__":
    engine = create_engine("postgresql://" + os.environ["POSTGRES"])
    cities = pd.read_sql(
        """SELECT * 
        FROM cities
        WHERE local_crs is not null AND code is not null""",
        con=engine,
    )

    cities = cities.sort_values(["id"])[["id", "code", "local_crs"]].to_dict("records")
    city_models = {
        city["code"]: DataQueryInterface(city["code"], city["local_crs"], city["id"])
        for city in cities
    }

    ready_for_metrics = [
        city for city, model in city_models.items() if pickle.loads(model.readiness)
    ]
    logger.warning(", ".join(ready_for_metrics) + " are ready for metrics.")

    MyService.ready_cities = ready_for_metrics

    t = ThreadedServer(
        MyService,
        port=18861,
        protocol_config={"allow_public_attrs": True, "allow_pickle": True},
    )
    print("starting")
    t.start()
