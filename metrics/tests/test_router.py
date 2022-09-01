import pytest

from tests.conf import testing_settings
from tests.geojson_example import CitiesPolygonForTrafficsCalculation
from metrics.app import schemas, enums


class TestTrafficsCalculation:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/pedastrian_walk_traffics"

    @pytest.mark.parametrize("city, geojson", [
        (enums.CitiesEnum.SAINT_PETERSBURG, CitiesPolygonForTrafficsCalculation.SAINT_PETERSBURG_INSIDE_GEOJSON),
        (enums.CitiesEnum.KRASNODAR, CitiesPolygonForTrafficsCalculation.KRASNODAR_INSIDE_GEOJSON),
        (enums.CitiesEnum.SEVASTOPOL, CitiesPolygonForTrafficsCalculation.SEVASTOPOL_INSIDE_GEOJSON),
    ])
    def test_pedastrian_walk_traffics_calculation(self, client, city, geojson):
        url = self.URL + "/pedastrian_walk_traffics_calculation"
        resp = client.post(url, json={"city": city, "geojson": geojson})

        assert resp.status_code == 200

    @pytest.mark.parametrize("city, geojson", [
        (enums.CitiesEnum.SAINT_PETERSBURG, CitiesPolygonForTrafficsCalculation.SAINT_PETERSBURG_OUTSIDE_GEOJSON),
        (enums.CitiesEnum.KRASNODAR, CitiesPolygonForTrafficsCalculation.KRASNODAR_OUTSIDE_GEOJSON),
        (enums.CitiesEnum.SEVASTOPOL, CitiesPolygonForTrafficsCalculation.SEVASTOPOL_OUTSIDE_GEOJSON),
    ])
    def test_400_error(self, client, city, geojson):
        url = self.URL + "/pedastrian_walk_traffics_calculation"
        resp = client.post(url, json={"city": city, "geojson": geojson})

        assert resp.status_code == 400


class TestVisibilityAnalysis:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/visibility_analysis"
    VIEWPOINTS = [  # random points in city bbox. latitude, longitude
        (enums.CitiesEnum.SAINT_PETERSBURG, 59.785982, 30.2971539),
        (enums.CitiesEnum.KRASNODAR, 45.0111502, 38.9100388),
        (enums.CitiesEnum.SEVASTOPOL, 44.7775737, 33.4171179),
    ]

    @pytest.mark.parametrize("view_distance", [700])
    @pytest.mark.parametrize("city, x_from, y_from", VIEWPOINTS)
    def test_visibility_analysis(self, client, city, x_from, y_from, view_distance):
        url = self.URL + "/visibility_analysis"
        params = {
            "city": city,
            "x_from": x_from,
            "y_from": y_from,
            "view_distance": view_distance,
        }

        resp = client.get(url, params=params)
        assert resp.status_code == 200


class TestWeightedVoronoi:  # fixme 4326
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/voronoi"
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
             "geometry": {"type": "Point", "coordinates": [3365424.7412466537, 8388892.177795848]}},
            {"type": "Feature", "properties": {"weight": 6.51134526170844},
             "geometry": {"type": "Point", "coordinates": [3366784.3993621334, 8390015.740133371]}},
            {"type": "Feature", "properties": {"weight": 4.7412358799581495},
             "geometry": {"type": "Point", "coordinates": [3366811.6122107455, 8384317.056265167]}}]
    }

    @pytest.mark.parametrize("city, geojson", [  # todo add geojson to KRASNODAR and SEVASTOPOL
        (enums.CitiesEnum.SAINT_PETERSBURG, SAINT_PETERSBURG_VORONOI_GEOJSON)
    ])
    def test_Weighted_voronoi_calculation(self, client, city, geojson):
        url = self.URL + "/Weighted_voronoi_calculation"
        data = {
            "city": city,
            "geojson": geojson,
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200


class TestBlocksClusterization:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/blocks_clusterization"
    RANDOM_SERVICE_TYPES = ["garbage_containers", "bakeries"]

    @pytest.mark.parametrize("clusters_number, service_types", [
        ("default", RANDOM_SERVICE_TYPES),
        (5, RANDOM_SERVICE_TYPES),  # 5 random value
    ])
    @pytest.mark.parametrize("city", enums.CitiesEnum)
    def test_get_blocks_calculations(self, client, city, clusters_number, service_types):
        url = self.URL + "/get_blocks"
        data = {
            "city": city,
            "param":
                {
                    "clusters_number": clusters_number,
                    "service_types": service_types,
                }
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("city", enums.CitiesEnum)
    @pytest.mark.parametrize("service_types, expected_status_code", [
        (RANDOM_SERVICE_TYPES, 200),
        ([], 422)
    ])
    def test_get_services_calculations_without_area(self, client, city, service_types, expected_status_code):
        url = self.URL + "/get_services"
        data = {
            "city": city,
            "param":
                {
                    "service_types": service_types,
                }
        }

        resp = client.post(url, json=data)
        assert resp.status_code == expected_status_code

    @pytest.mark.parametrize("city, area_type, area_id", [
        (enums.CitiesEnum.SAINT_PETERSBURG, "block", 856),
        (enums.CitiesEnum.KRASNODAR, "block", 7259),
        (enums.CitiesEnum.SEVASTOPOL, "block", 14518),
    ])
    def test_get_services_calculations(self, client, city, area_type, area_id):
        url = self.URL + "/get_services"
        data = {
            "city": city,
            "param":
                {
                    "area": {area_type: area_id},
                }
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("clusters_number, service_types", [
        ("default", RANDOM_SERVICE_TYPES),
        (5, RANDOM_SERVICE_TYPES),  # 5 random value
    ])
    @pytest.mark.parametrize("city", enums.CitiesEnum)
    def test_get_dendrogram(self, client, city, clusters_number, service_types):
        url = self.URL + "/get_dendrogram"
        data = {
            "city": city,
            "param":
                {
                    "clusters_number": clusters_number,
                    "service_types": service_types,
                }
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "image/png"


class TestServicesClusterization:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/services_clusterization"
    RANDOM_SERVICE_TYPES = ["garbage_containers", "bakeries"]

    DISTRICTS = [
        (enums.CitiesEnum.SAINT_PETERSBURG, "district", 48),
        (enums.CitiesEnum.KRASNODAR, "district", 67),
        (enums.CitiesEnum.SEVASTOPOL, "district", 86),
    ]
    MUNICIPALITIES = [
        (enums.CitiesEnum.SAINT_PETERSBURG, "mo", 95),
        (enums.CitiesEnum.KRASNODAR, "mo", 113),
        (enums.CitiesEnum.SEVASTOPOL, "mo", 126),
    ]

    @pytest.mark.parametrize("city", enums.CitiesEnum)
    @pytest.mark.parametrize("service_types, expected_status_code", [
        (RANDOM_SERVICE_TYPES, 200),
        ([], 422),
    ])
    def test_get_services_without_area(self, client, city, service_types, expected_status_code):
        url = self.URL + "/get_services"
        data = {
            "city": city,
            "param": {
                "service_types": service_types,
            }
        }

        resp = client.post(url, json=data)
        assert resp.status_code == expected_status_code

    @pytest.mark.parametrize("city, area_type, area_id", [*DISTRICTS, *MUNICIPALITIES])
    def test_get_services(self, client, city, area_type, area_id):
        url = self.URL + "/get_services"
        data = {
            "city": city,
            "param":
                {
                    "area": {area_type: area_id},
                }
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("city, area_type, area_id", [*DISTRICTS, *MUNICIPALITIES])
    @pytest.mark.parametrize("condition", [
        {"default": "default"},
        {"distance": 1000},  # random distance value
        {"maxclust": 2},  # random maxclust value
    ])
    @pytest.mark.parametrize("n_std", [
        2,  # default value
        10,  # random value
    ])
    @pytest.mark.parametrize("service_types", [RANDOM_SERVICE_TYPES])
    def test_get_services_clusterization(self, client, city, area_type, area_id, condition, n_std, service_types):
        url = self.URL + "/get_clusters_polygons"
        data = {
            "city": city,
            "param":
                {
                    "area": {area_type: area_id},
                    "service_types": service_types,
                    "condition": condition,
                    "n_std": n_std,
                }
        }
        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("city", enums.CitiesEnum)
    @pytest.mark.parametrize("service_types", [RANDOM_SERVICE_TYPES])
    def test_get_services_clusterization_without_area(self, client, city, service_types):
        url = self.URL + "/get_clusters_polygons"
        data = {
            "city": city,
            "param": {
                "service_types": service_types,
            }
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("city", enums.CitiesEnum)
    @pytest.mark.parametrize("service_types", [
        ["does_not_exists_service"],
    ])
    def test_get_services_clusterization_without_objects_to_cluster(self, client, city, service_types):
        """ Тестирование случаев, когда нет сервисов для кластеризации. """
        url = self.URL + "/get_clusters_polygons"
        data = {
            "city": city,
            "param": {
                "service_types": service_types,
            }
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 400

        error_msg = "Not enough objects to cluster"
        assert error_msg in resp.text


@pytest.mark.skip(reason="Not implemented test")
def test_spacematrix_objects():
    ...


class TestMobilityAnalysisIsochrones:
    """ Проверка метрики доступности. """
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/mobility_analysis/isochrones"
    CITIES_FROM_POINTS = [  # random points in city bbox. latitude, longitude
        (enums.CitiesEnum.SAINT_PETERSBURG, 59.9386300, 30.3141300),
        (enums.CitiesEnum.KRASNODAR, 45.0448400, 38.9760300),
        (enums.CitiesEnum.SEVASTOPOL, 44.5888300, 33.5224000)
    ]

    @pytest.mark.parametrize("travel_type", [
        enums.MobilityAnalysisIsochronesTravelTypeEnum.PUBLIC_TRANSPORT,
    ])
    @pytest.mark.parametrize("weight_type, weight_value", [
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.TIME, 1)
    ])
    @pytest.mark.parametrize("city, x_from, y_from", CITIES_FROM_POINTS)
    def test_public_transport_travel_type(
            self, client, city, x_from, y_from, weight_type, weight_value, travel_type
    ):
        """ Проверка вычисления изохрон для общественного транспорта. """
        params = dict(
            city=city,
            travel_type=travel_type,
            weight_type=weight_type,
            weight_value=weight_value,
            x_from=x_from,
            y_from=y_from,
        )

        url = self.URL

        resp = client.get(url, params=params)
        assert resp.status_code == 200

    @pytest.mark.parametrize("travel_type", [
        enums.MobilityAnalysisIsochronesTravelTypeEnum.PUBLIC_TRANSPORT,
    ])
    @pytest.mark.parametrize("weight_type, weight_value", [
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.METER, 100)
    ])
    @pytest.mark.parametrize("city, x_from, y_from", CITIES_FROM_POINTS)
    def test_not_implemented_public_transport_travel_type_with_meter(self, client, city, x_from, y_from,
                                                                     weight_type, weight_value, travel_type):
        """ Проверка вычисления изохрон для общественного транспорта c нереализованным типом весов графа - метры. """
        params = dict(
            city=city,
            travel_type=travel_type,
            weight_type=weight_type,
            weight_value=weight_value,
            x_from=x_from,
            y_from=y_from,
        )

        url = self.URL

        resp = client.get(url, params=params)
        assert resp.status_code == 422

    @pytest.mark.parametrize("travel_type", [
        enums.MobilityAnalysisIsochronesTravelTypeEnum.DRIVE,
        enums.MobilityAnalysisIsochronesTravelTypeEnum.WALK,
    ])
    @pytest.mark.parametrize("weight_type, weight_value", [
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.TIME, 10),
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.METER, 100),  # todo check linestring
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.METER, 250)
    ])
    @pytest.mark.parametrize("city, x_from, y_from", CITIES_FROM_POINTS)
    def test_drive_and_walk_travel_type(self, client, city, x_from, y_from, weight_type, weight_value, travel_type):
        """ Проверка вычисления изохрон для личного транспорта. """
        params = dict(
            city=city,
            travel_type=travel_type,
            weight_type=weight_type,
            weight_value=weight_value,
            x_from=x_from,
            y_from=y_from,
        )

        url = self.URL

        resp = client.get(url, params=params)
        assert resp.status_code == 200


class TestDiversity:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/diversity"
    RANDOM_SERVICE_TYPES = ["garbage_containers", "bakeries"]

    @pytest.mark.parametrize("service_type", RANDOM_SERVICE_TYPES)
    def test_get_diversity(self, client, service_type):
        url = self.URL + "/diversity"
        params = {
            "service_type": service_type,
        }

        resp = client.get(url, params=params)
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="500 ошибка если указать некорректный сервис. Сделать валидацию сервисов")
    def test_get_diversity_does_not_exists_service_type(self, client):
        url = self.URL + "/diversity"
        params = {
            "service_type": "cafe",
        }

        resp = client.get(url, params=params)
        assert resp.status_code == 422


@pytest.mark.skip(reason="Not implemented test")
def test_get_provision():
    ...


@pytest.mark.skip(reason="Not implemented test")
def test_get_provision_info():
    ...


@pytest.mark.skip(reason="Not implemented test")
def test_get_wellbeing():
    ...


class TestBCAM:
    class TestMobilityAnalysisRoutes:
        """ Проверка метрики доступности. """
        URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/mobility_analysis/routes"
        FROM_AND_TO_POINTS = [  # random points in city bbox. latitude, longitude, latitude, longitude
            (enums.CitiesEnum.SAINT_PETERSBURG, 59.785982, 30.2971539, 60.0863286, 29.4933082),
            (enums.CitiesEnum.KRASNODAR, 45.0111502, 38.9100388, 45.0796982, 38.9356331),
            (enums.CitiesEnum.SEVASTOPOL, 44.7775737, 33.4171179, 44.7581441, 33.7148713),
        ]

        POINTS_WITHOUT_PATH_BETWEEN = [
            (
                enums.CitiesEnum.SAINT_PETERSBURG, enums.MobilityAnalysisRoutesTravelTypeEnum.DRIVE,
                59.6950839, 30.2527172, 60.0936915, 30.653769)
        ]

        @pytest.mark.parametrize("travel_type", enums.MobilityAnalysisRoutesTravelTypeEnum)
        @pytest.mark.parametrize("city,  x_from, y_from, x_to, y_to", FROM_AND_TO_POINTS)
        def test_mobility_analysis_routes(self, client, city, travel_type, x_from, y_from, x_to, y_to):
            """ Проверка нахождения маршрута между двумя точками для всех видов передвижений. """
            params = dict(
                city=city,
                travel_type=travel_type,
                x_from=x_from,
                y_from=y_from,
                x_to=x_to,
                y_to=y_to,
            )
            url = self.URL
            resp = client.get(url, params=params)
            assert resp.status_code == 200

            schemas.MobilityAnalysisRoutesOut(**resp.json())

        @pytest.mark.parametrize("travel_type", [
            pytest.param("invalid_travel_type", marks=pytest.mark.xfail(reasone="Need input validate scheme"))
        ])
        @pytest.mark.parametrize("city,  x_from, y_from, x_to, y_to", [FROM_AND_TO_POINTS[0]])  # only for one city
        def test_invalid_travel_type(self, client, city, travel_type, x_from, y_from, x_to, y_to):
            """ Проверка корректности указанного типа перемещения."""
            url = self.URL
            params = dict(
                city=city,
                travel_type=travel_type,
                x_from=x_from,
                y_from=y_from,
                x_to=x_to,
                y_to=y_to,
            )
            resp = client.get(url, params=params)
            assert resp.status_code == 422

        @pytest.mark.parametrize("city, travel_type, x_from, y_from, x_to, y_to", POINTS_WITHOUT_PATH_BETWEEN)
        def test_points_without_path_between(self, client, city, travel_type, x_from, y_from, x_to, y_to):
            """ Проверка ошибки, если нет пути между двумя указанными точками. """
            params = dict(
                city=city,
                travel_type=travel_type,
                x_from=x_from,
                y_from=y_from,
                x_to=x_to,
                y_to=y_to,
            )
            url = self.URL
            resp = client.get(url, params=params)
            assert resp.status_code == 422

            error_msg = "Path between given points absents"
            assert error_msg in resp.text


class TestCMM:
    class TestConnectivityCalculations:
        URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/connectivity_calculations"

        def test_connectivity_viewer(self, client, random_spb_municipality):
            url = self.URL + "/connectivity_viewer"

            feature_properties = {
                "id": 1  # todo fake id?
            }

            data = {
                "type": "FeatureCollection",
                "name": "test_area",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": [
                    {
                        "type": "Feature",
                        "properties": feature_properties,
                        "geometry": random_spb_municipality.dict()
                    }
                ],
            }
            resp = client.post(url, json=data)

            assert resp.status_code == 200

    @pytest.mark.xfail(reason="Тест падает после запуска тест кейсов TestConnectivityCalculations.")
    class TestHouseLocation:
        URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/house_location"
        RANDOM_SERVICE_TYPES = ["garbage_containers", "bakeries"]

        @pytest.mark.parametrize("area, floors, population, service_types", [
            (2000, 5, 300, RANDOM_SERVICE_TYPES)
        ])
        def test_House_location_calculations(self, client, area, floors, population, service_types):
            url = self.URL + "/house_location"
            data = {
                "area": area,
                "floors": floors,
                "population": population,
                "service_types": service_types
            }

            resp = client.post(url, json=data)
            assert resp.status_code == 200

        @pytest.mark.parametrize("block_id, population, service_types", [
            (856, 300, RANDOM_SERVICE_TYPES)
        ])
        def test_block_provision_calculations(self, client, block_id, population, service_types):
            url = self.URL + "/block_provision"
            data = {
                "block_id": block_id,
                "population": population,
                "service_types": service_types
            }

            resp = client.post(url, json=data)
            assert resp.status_code == 200

        def test_block_provision_calculations_without_services_in_block(self):
            """

            Тестовый случай когда внутри квартала нет сервисов.
            Ожидается 400 ошибка.
            """

    @pytest.mark.xfail(reason="Тест падает после запуска тест кейсов TestConnectivityCalculations.")
    class TestServicesLocation:
        URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/service_location"

        @pytest.mark.parametrize("user_service_choice, user_unit_square_min, user_unit_square_max", [
            ("Кафе/столовая", 10, 50)
        ])
        def test_service_location(self, client, user_service_choice, user_unit_square_min, user_unit_square_max):
            url = self.URL + "/service_location"
            data = {
                "user_service_choice": user_service_choice,
                "user_unit_square_min": user_unit_square_min,
                "user_unit_square_max": user_unit_square_max
            }

            resp = client.post(url, json=data)
            assert resp.status_code == 200

    class TestHouseSelection:
        URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/house_selection"

        RANDOM_SIGNIFICANT_SERVICES = ["Урна", "Автозаправка"]
        RANDOM_UNSIGNIFICANT_SERVICES = ["Кафе/столовая"]
        RANDOM_SOCIAL_GROUP = "Дети младшего школьного возраста (7-11)"

        @pytest.mark.xfail(reason="'InterfaceCityInformationModel' object has no attribute 'Social_groups'")
        def test_get_social_groups_list(self, client):
            url = self.URL + "/social_groups_list"

            resp = client.get(url)
            assert resp.status_code == 200

        @pytest.mark.parametrize(
            "user_selected_significant_services, user_selected_unsignificant_services, \
            user_social_group_selection, user_price_preferences", [
                (RANDOM_SIGNIFICANT_SERVICES, RANDOM_UNSIGNIFICANT_SERVICES, RANDOM_SOCIAL_GROUP, [20000, 30000]),
            ])
        def test_House_selection_calculations(
                self, client,
                user_selected_significant_services, user_selected_unsignificant_services,
                user_social_group_selection, user_price_preferences
        ):
            url = self.URL + "/House_selection"
            data = {
                "user_selected_significant_services": user_selected_significant_services,
                "user_selected_unsignificant_services": user_selected_unsignificant_services,
                "user_social_group_selection": user_social_group_selection,
                "user_price_preferences": user_price_preferences,
            }
            resp = client.post(url, json=data)
            assert resp.status_code == 200


@pytest.mark.xfail(reason="'InterfaceCityInformationModel' object has no attribute 'get_instagram_data'")
class TestInstagram:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/instagram_concentration"

    @pytest.mark.parametrize("year", [2018, 2019, 2020])  # todo replace on enum
    @pytest.mark.parametrize("season", ["white_nights", "summer", "winter", "spring+autumn"])  # todo replace on enum
    @pytest.mark.parametrize("day_time", ["dark", "light"])  # todo replace on enum
    def test_instagram_get_squares(self, client, year, season, day_time):
        url = self.URL + "/get_squares"
        params = {
            "year": year,
            "season": season,
            "day_time": day_time,
        }

        resp = client.get(url, params=params)
        assert resp.status_code == 200
