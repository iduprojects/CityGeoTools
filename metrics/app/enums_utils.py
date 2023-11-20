import rpyc
import os
from enum import Enum
from typing import List, Dict


def get_rpyc_addr_port():
    DEFAULT_RPYC_PORT = 18861
    rpyc_server = os.environ["RPYC_SERVER"]
    address, port = (
        rpyc_server.split(":")
        if ":" in rpyc_server
        else (rpyc_server, DEFAULT_RPYC_PORT)
    )

    return address, port


def connect_to_rpyc(address, port):
    return rpyc.connect(address, port)


def get_ready_cities_from_rpyc(conn) -> List[str]:
    return conn.root.ready_cities


def get_ready_for_metrics_cities() -> List[str]:
    addr, port = get_rpyc_addr_port()
    conn = connect_to_rpyc(addr, port)
    cities_list = get_ready_cities_from_rpyc(conn)
    return cities_list


def create_dict_of_ready_cities(base_cities, ready_cities) -> Dict[str, str]:
    ready_cities_dict = {}
    base_cities_names = base_cities.values()
    for city in ready_cities:
        if city not in base_cities_names:
            ready_cities_dict[city.upper()] = city
    return ready_cities_dict


def get_all_ready_cities_dict() -> Dict[str, str]:
    BASE_CITIES = {
        "SAINT_PETERSBURG": "saint-petersburg",
        "KRASNODAR": "krasnodar",
        "SEVASTOPOL": "sevastopol",
    }

    ready_cities = get_ready_for_metrics_cities()
    all_cities = create_dict_of_ready_cities(BASE_CITIES, ready_cities)
    all_cities.update(BASE_CITIES)

    return all_cities


def add_enums_of_ready_for_metrics_cities() -> Enum:
    all_cities = get_all_ready_cities_dict()
    return Enum("CitiesEnum", all_cities)
