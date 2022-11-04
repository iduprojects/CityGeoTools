from data_classes.InterfaceCityInformationModel import DataQueryInterface

cities_name = {"saint-petersburg": "Санкт-Петербург",
               "krasnodar": "Краснодар",
               "sevastopol": "Севастополь"}

cities_db_id = {"saint-petersburg": 1,
                "krasnodar": 2,
                "sevastopol": 5}

cities_crs = {"saint-petersburg": 32636,
              "krasnodar": 32637,
              "sevastopol": 32636}

cities_model = {name: DataQueryInterface(name, cities_crs[name], cities_db_id[name]) 
                for name in cities_name}