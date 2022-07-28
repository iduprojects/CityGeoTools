import os
from Data.CityInformationModel import CityInformationModel

path = os.getcwd().split("/Data")[0]

cities_name = {"Saint_Petersburg": "Санкт-Петербург",
               "Krasnodar": "Краснодар",
               "Sevastopol": "Севастополь"}

cities_db_id = {"Saint_Petersburg": 1,
                "Krasnodar": 2,
                "Sevastopol": 5}

cities_crs = {"Saint_Petersburg": 32636,
              "Krasnodar": 32637,
              "Sevastopol": 32636}

cities_model = {name: CityInformationModel(name, cities_crs["name"], cities_db_id["name"]) for name in cities_name}

cities_metrics = {"Saint_Petersburg": ["connectivity_calculations", "pedastrian_walk_traffics",
                                       "mobility_analysis", "Visibility_analysis", "voronoi",
                                       "instagram_concentration", "house_location",
                                       "blocks_clusterization", "services_clusterization", "diversity"
                                       "house_selection", "service_location", "spacematrix", "provision"],

                  "Krasnodar": ["mobility_analysis", "Visibility_analysis", "voronoi",
                                "blocks_clusterization", "services_clusterization"],
                  "Sevastopol": ["mobility_analysis", "Visibility_analysis", "voronoi",
                                 "blocks_clusterization"]}
