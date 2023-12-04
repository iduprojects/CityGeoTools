import os
import pandas as pd

from sqlalchemy import create_engine
from data.CityInformationModel import CityInformationModel

rpyc_server = os.environ["RPYC_SERVER"]
postgres_con = "postgresql://" + os.environ["POSTGRES"]
engine = create_engine(postgres_con)
address, port = rpyc_server.split(":") if ":" in rpyc_server else (rpyc_server, 18861)

cities = pd.read_sql(
    """SELECT * FROM cities
    WHERE local_crs is not null AND code is not null and id in (1,2,5)""", 
    con=engine)

city_names = cities.set_index("code")["name"].to_dict()

cities = cities.sort_values(["id"])[["id", "code", "local_crs"]].to_dict("records")
city_models = {city["code"]: CityInformationModel(
    city_name=city["code"], city_crs=city["local_crs"], cities_db_id=city["id"], mode="general_mode", 
    postgres_con=postgres_con, rpyc_adr=address, rpyc_port=port
    ) for city in cities}