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

    def get_city_model_attr(self, city_name, atr_name):
        print(city_name, datetime.datetime.now(), atr_name)
        return getattr(self.city_models[city_name], atr_name)
    
    def get_provisions(self,city_name,atr_name, chunk_num):
        print(city_name, datetime.datetime.now(), atr_name, chunk_num)
        return getattr(self.city_models[city_name], atr_name)[chunk_num]


if __name__ == "__main__":

    engine = create_engine("postgresql://" + os.environ["POSTGRES"])
    cities = pd.read_sql(
        """SELECT * 
        FROM cities
        WHERE local_crs is not null AND name_en is not null""", con=engine).sort_values(["id"])

    cities = cities[["id", "name_en", "local_crs"]].to_dict("records")
    city_models = {
        city["name_en"]: DataQueryInterface(city["name_en"], city["local_crs"], city["id"]) for city in cities
        }

    ready_for_metrics = [city for city, model in city_models.items() if pickle.loads(model.readiness)]
    logger.warning(", ".join(ready_for_metrics) + " are ready for metrics.")

    t = ThreadedServer(MyService, port=18861
                                , protocol_config={"allow_public_attrs": True, 
                                                   "allow_pickle": True})
    print('starting')
    t.start()
