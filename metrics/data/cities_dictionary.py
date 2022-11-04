import os
from data.CityInformationModel import CityInformationModel

path = os.getcwd().split("/Data")[0]

cities_name = {"saint-petersburg": "Санкт-Петербург",
               "krasnodar": "Краснодар",
               "sevastopol": "Севастополь"}

cities_db_id = {"saint-petersburg": 1,
                "krasnodar": 2,
                "sevastopol": 5}

cities_crs = {"saint-petersburg": 32636,
              "krasnodar": 32637,
              "sevastopol": 32636}

cities_model = {name: CityInformationModel(name, cities_crs[name], cities_db_id[name], mode="general_mode") 
                for name in cities_name}
