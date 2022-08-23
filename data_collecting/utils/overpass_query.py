import requests
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, Point
from json import JSONDecodeError
from utils.transform import *


def overpass_get_boundary(osm_id):
    
    overpass_url = "http://lz4.overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
            (
              relation({osm_id});
            );
    out geom;
    """
    result = requests.get(overpass_url, params={'data': overpass_query})
    json_result = handle_overpass_errors(result)
    
    return json_result


def overpass_get_routes(osm_id, public_transport_type):
    
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
    result = requests.get(overpass_url, params={'data': overpass_query})
    json_result = handle_overpass_errors(result)["elements"]
    
    return pd.DataFrame(json_result)


def handle_overpass_errors(result):

    try:
        return result.json()
    except JSONDecodeError as e:
        raise JSONDecodeError(
          msg=f"""Something went wrong with Overpass API when JSON was parsed. Check the query and to send it later.
                 {e.msg}""", doc=e.doc, pos=e.pos)


def parse_overpass_route_response(loc, city_crs):
    
    route = pd.DataFrame(loc['members'])
    ways = route[route['type'] == 'way']
    if len(ways) > 0:
        ways = ways['geometry'].reset_index(drop = True)
        ways = ways.apply(lambda x: pd.DataFrame(x))
        ways = ways.apply(lambda x: LineString(list(zip(x['lon'], x['lat']))))
        ways = gpd.GeoDataFrame(ways.rename("geometry")).set_crs(4326).to_crs(city_crs)
        
        # fix topological errors and then make LineString from MultiLineString
        ways = get_linestring(ways)
    else:
        ways = None

    if "node" in route["type"].unique():
        platforms = route[route['type'] == 'node'][["lat", "lon"]].reset_index(drop = True)
        platforms = platforms.apply(lambda x: Point(x["lon"], x["lat"]), axis=1)
    else:
        platforms = None
        
    return pd.Series({"way": ways, "platforms": platforms})
