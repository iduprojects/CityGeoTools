import requests
import pandas as pd
import geopandas as gpd
import time

from shapely.geometry import LineString, Point
from json import JSONDecodeError
from data_collecting.utils.transform import *


def get_boundary(osm_id):
    
    overpass_url = "http://lz4.overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
            (
              relation({osm_id});
            );
    out geom;
    """
    result = requests.get(overpass_url, params={'data': overpass_query}, timeout=600)
    json_result = result.json()
    
    return json_result


def get_routes(osm_id, public_transport_type):
    
    overpass_url = "http://lz4.overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
            (
                relation({osm_id});
            );map_to_area;
            (
                relation(area)['route'='{public_transport_type}'];
            );
    out geom;
    """
    result = requests.get(overpass_url, params={'data': overpass_query}, timeout=600)
    json_result = result.json()["elements"]
    
    return pd.DataFrame(json_result)


def overpass_query(func, *args, attempts=5):

    for i in range(attempts):
        try:
            return func(*args)
        except JSONDecodeError:
            print("Another attempt to get response from Overpass API...")
            time.sleep(20)
            continue
    
    raise SystemError(
    """Something went wrong with Overpass API when JSON was parsed. Check the query and to send it later.""")


def parse_overpass_route_response(loc, city_crs, boundary):
    
    route = pd.DataFrame(loc['members'])
    ways = route[route['type'] == 'way']
    if len(ways) > 0:
        ways = ways['geometry'].reset_index(drop = True)
        ways = ways.apply(lambda x: pd.DataFrame(x))
        ways = ways.apply(lambda x: LineString(list(zip(x['lon'], x['lat']))))
        ways = gpd.GeoDataFrame(ways.rename("geometry")).set_crs(4326)
        if ways.within(boundary).all():
            # fix topological errors and then make LineString from MultiLineString
            ways = get_linestring(ways.to_crs(city_crs))
        else:
            ways = None
    else:
        ways = None

    if "node" in route["type"].unique():
        platforms = route[route['type'] == 'node'][["lat", "lon"]].reset_index(drop = True)
        platforms = platforms.apply(lambda x: Point(x["lon"], x["lat"]), axis=1)
    else:
        platforms = None
        
    return pd.Series({"way": ways, "platforms": platforms})
