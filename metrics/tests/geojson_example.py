"""Примеры GEOJSON для тестирования."""


class CitiesPolygonForTrafficsCalculation:
    SAINT_PETERSBURG_INSIDE_GEOJSON = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "Polygon",
                          "coordinates": [
                              [[30.31498650660696, 59.93638028149156], [30.311885872985744, 59.93523889144893],
                               [30.313581029083153, 59.9341243829608], [30.31658510317982, 59.935217355409094],
                               [30.31498650660696, 59.93638028149156]]]
                          }}]
    }

    SAINT_PETERSBURG_OUTSIDE_GEOJSON = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [
                                28.992919921875004,
                                61.03169171684717
                            ],
                            [
                                28.7841796875,
                                60.77525532466672
                            ],
                            [
                                29.33349609375,
                                60.6301017662667
                            ],
                            [
                                29.838867187500004,
                                60.91441435497479
                            ],
                            [
                                28.992919921875004,
                                61.03169171684717
                            ]
                        ]
                    ]
                }
            }
        ]
    }

    KRASNODAR_INSIDE_GEOJSON = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[38.958311612232684, 45.04137098808761], [38.95773225508545, 45.039697541830854],
                         [38.95996385298586, 45.039378060125934], [38.960435921772486, 45.041081941800066],
                         [38.958311612232684, 45.04137098808761]]
                    ]
                }
            }
        ]
    }

    KRASNODAR_OUTSIDE_GEOJSON = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[37.668111614501726, 44.14852725957535], [37.635152630126726, 43.71980691882187],
                         [38.228414348876726, 43.926616327734955], [37.668111614501726, 44.14852725957535]]
                    ]
                }
            }
        ]
    }

    SEVASTOPOL_INSIDE_GEOJSON = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[33.52756071429436, 44.56476523653583], [33.52867651324456, 44.562893728399125],
                         [33.53127289157099, 44.56370676806933], [33.52981376986687, 44.56573164733193],
                         [33.52756071429436, 44.56476523653583]]
                    ]
                }
            }
        ]
    }

    SEVASTOPOL_OUTSIDE_GEOJSON = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[33.415349133157704, 44.53495890620129], [33.40436280503272, 44.52120555877456],
                         [33.4266787840366, 44.521942428130714], [33.415349133157704, 44.53495890620129]]
                    ]

                }
            }
        ]
    }


SAINT_PETERSBURG_VORONOI_GEOJSON = {
    "type": "FeatureCollection",
    "name": "test",
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:EPSG::3857"
        }
    },
    "features": [
        {"type": "Feature", "properties": {"weight": 5.330623275960242},
         "geometry": {"type": "Point", "coordinates": [30.31498650660696, 59.93638028149156]}},
        {"type": "Feature", "properties": {"weight": 6.51134526170844},
         "geometry": {"type": "Point", "coordinates": [30.311885872985744, 59.93523889144893]}},
        {"type": "Feature", "properties": {"weight": 4.7412358799581495},
         "geometry": {"type": "Point", "coordinates": [30.313581029083153, 59.9341243829608]}},
        {"type": "Feature", "properties": {"weight": 3.9958149574123587},
         "geometry": {"type": "Point", "coordinates": [30.31658510317982, 59.935217355409094]}},
    ]
}


SAINT_PETERSBURG_DIVERSITY_GEOJSON = {

                    "type": "FeatureCollection",
                    "name": "test_area",
                    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                    "features": [
                        {"type": "Feature", "properties": {},
                         "geometry": {"type": "Polygon",
                                      "coordinates": [[
                                        [30.310845, 59.945431], 
                                        [30.189368, 59.969446], 
                                        [30.165860, 59.932500], 
                                        [30.263008, 59.917023], 
                                        [30.274934, 59.930412], 
                                        [30.314616, 59.944886],
                                        [30.310845, 59.945431]
                                        ]
]}}]}