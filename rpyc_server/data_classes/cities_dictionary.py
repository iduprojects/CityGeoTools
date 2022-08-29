from data_classes.InterfaceCityInformationModel import DataQueryInterface

cities_name = {"Saint_Petersburg": "Санкт-Петербург",
               "Krasnodar": "Краснодар",
               "Sevastopol": "Севастополь"}

cities_db_id = {"Saint_Petersburg": 1,
                "Krasnodar": 2,
                "Sevastopol": 5}

cities_crs = {"Saint_Petersburg": 32636,
              "Krasnodar": 32637,
              "Sevastopol": 32636}

cities_model = {name: DataQueryInterface(name, cities_crs[name], cities_db_id[name]) 
                for name in cities_name}