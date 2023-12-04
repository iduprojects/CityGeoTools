provisions_tests_kinders = {
    "type": "FeatureCollection",
    "name": "kindergartens_test",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [
        {
            "type": "Feature",
            "properties": {
                "building_id": 1.0,
                "id": 1,
                "city_service_type": None,
                "city_service_type_id": None,
                "service_code": None,
                "service_name": None,
                "address": None,
                "capacity": 1000,      
                "capacity_left":1000,
                "carried_capacity_within":0,
                "carried_capacity_without":0,
                "block_id": None,
                "administrative_unit_id": None,
                "municipality_id": None,
                "x": None,
                "y": None,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [30.210594413448785, 59.987875558496668],
            },
        },
    ],
}

provisions_tests_kinders_houses = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "id": 15448.0,
                "house_id": 15448.0,
                "basement_area": 243.263,
                "is_living": True,
                "living_area": 158.6,
                "population": 3.0,
                "storeys_count": 1.0,
                "functional_object_id": 130007.0,
                "address": "Санкт-Петербург, Кольцевая, 15",
                "administrative_unit_id": 48.0,
                "municipality_id": 95.0,
                "block_id": None,
                "x": 332781.39653516706,
                "y": 6656302.2062337566,
                "kindergartens_demand": 345,
                "demand_left": 2,
                "kindergartens_provison_value":0.0,
                "kindergartens_demand_left": 0.0,
                "kindergartens_supplyed_demands_within": 2.0,
                "kindergartens_supplyed_demands_without": 0.0,
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [30.0003209, 60.0098016],
                            [30.0003266, 60.009932199999987],
                            [30.000626, 60.009929],
                            [30.0006204, 60.0097983],
                            [30.0003209, 60.0098016],
                        ]
                    ]
                ],
            },
        },
    ],
}

provisions_tests_kinders_provisions = {
    "house_id": 131560,
    "demand": 19.0,
    "service_id": 145813,
    "kindergartens": provisions_tests_kinders
}
