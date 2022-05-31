from asyncio.log import logger
from dataclasses import dataclass
import rpyc
from rpyc.utils.server import ThreadedServer

from data_classes.InterfaceCityInformationModel import InterfaceCityInformationModel
from data_classes.Cities_dictionary import cities_model, cities_crs, cities_metrics, cities_name

import pickle

import datetime


class MyService(rpyc.Service):

    def get_city_model_attr(self, city_name,atr_name):
        print(city_name, datetime.datetime.now(), atr_name)
        return getattr(cities_model[city_name], atr_name)
    
    def get_provisions(self,city_name,atr_name, chunk_num):
        print(city_name, datetime.datetime.now(), atr_name, chunk_num)
        return getattr(cities_model[city_name], atr_name)[chunk_num]

if __name__ == "__main__":

    t = ThreadedServer(MyService, port=18861
                                , protocol_config={"allow_public_attrs": True, 
                                                   "allow_pickle":True})
    print('starting')
    t.start()