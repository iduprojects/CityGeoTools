import random

import requests
import pytest
from geojson_pydantic import Polygon

from tests.conf import testing_settings


@pytest.fixture
def outer_client():
    """Client for make requests to outer apps"""
    return requests


@pytest.fixture
def client(outer_client):
    """Client for make requests to testing app"""
    return outer_client


@pytest.fixture()
def random_spb_municipality(outer_client) -> Polygon:
    """Get random spb municipality from outer API. """
    data = requests.get(testing_settings.SPB_MUNICIPALITIES, timeout=600).json()
    random_municipality = random.choice(data)
    polygon = random_municipality["geometry"]["coordinates"][0]  # From multipolygon
    return Polygon(coordinates=polygon)
