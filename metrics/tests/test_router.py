import pytest

from tests.conf import testing_settings
from tests.geojson_example import CitiesPolygonForTrafficsCalculation, SAINT_PETERSBURG_VORONOI_GEOJSON
from app import enums
from tests import provision_geojson_examples

MUNICIPALITIES = [
    (enums.CitiesEnum.SAINT_PETERSBURG, enums.TerritorialEnum.MUNICIPALITY, 95),
    (enums.CitiesEnum.KRASNODAR, enums.TerritorialEnum.MUNICIPALITY, 113),
    (enums.CitiesEnum.SEVASTOPOL, enums.TerritorialEnum.MUNICIPALITY, 126),
]

BLOCKS = [
    (enums.CitiesEnum.SAINT_PETERSBURG, enums.TerritorialEnum.BLOCK, 2800),
    (enums.CitiesEnum.KRASNODAR, enums.TerritorialEnum.BLOCK, 7034),
    (enums.CitiesEnum.SEVASTOPOL, enums.TerritorialEnum.BLOCK, 15020),
]

ADMINISTRATIVE_UNITS = [
    (enums.CitiesEnum.SAINT_PETERSBURG, enums.TerritorialEnum.ADMINISTRATIVE_UNIT, 59),
    (enums.CitiesEnum.KRASNODAR, enums.TerritorialEnum.ADMINISTRATIVE_UNIT, 69),
    (enums.CitiesEnum.SEVASTOPOL, enums.TerritorialEnum.ADMINISTRATIVE_UNIT, 137),
]


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
        """Проверка территорий, в которые не попадают жилые дома."""
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


class TestWeightedVoronoi:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/voronoi"

    @pytest.mark.parametrize("city, geojson", [
        (enums.CitiesEnum.SAINT_PETERSBURG, SAINT_PETERSBURG_VORONOI_GEOJSON)
    ])
    def test_weighted_voronoi_calculation(self, client, city, geojson):
        url = self.URL + "/weighted_voronoi_calculation"
        data = {
            "city": city,
            "geojson": geojson,
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200


class TestBlocksClusterization:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/blocks_clusterization"
    RANDOM_SERVICE_TYPES = ["garbage_containers", "bakeries"]
    RANDOM_CLUSTER_NUMBER = 5
    DEFAULT_CLUSTER_NUMBER = None

    @pytest.mark.parametrize("clusters_number", [DEFAULT_CLUSTER_NUMBER, RANDOM_CLUSTER_NUMBER])
    @pytest.mark.parametrize("city, geojson", [
        (enums.CitiesEnum.SAINT_PETERSBURG, CitiesPolygonForTrafficsCalculation.SAINT_PETERSBURG_INSIDE_GEOJSON),
    ])
    def test_get_blocks_calculations(self, client, city, clusters_number, geojson):
        url = self.URL + "/get_blocks"
        data = {
            "city": city,
            "clusters_number": clusters_number,
            "service_types": self.RANDOM_SERVICE_TYPES,
            "geojson": geojson,
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("city, geojson", [
        (enums.CitiesEnum.SAINT_PETERSBURG, CitiesPolygonForTrafficsCalculation.SAINT_PETERSBURG_INSIDE_GEOJSON),
    ])
    def test_get_dendrogram(self, client, city, geojson):
        url = self.URL + "/get_dendrogram"
        data = {
            "city": city,
            "clusters_number": self.RANDOM_CLUSTER_NUMBER,
            "service_types": self.RANDOM_SERVICE_TYPES,
            "geojson": geojson,
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "image/png"


class TestServicesClusterization:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/services_clusterization"
    RANDOM_SERVICE_TYPES = ["garbage_containers", "bakeries", "restaurants", "fastfoods"]

    @pytest.mark.parametrize("city", enums.CitiesEnum)
    @pytest.mark.parametrize("condition", enums.ClusterizationConditionsEnum)
    def test_get_services_clusterization(self, client, city, condition):
        """Запрос с обязательными полями и значениями по умолчанию. """
        url = self.URL + "/get_clusters_polygons"
        data = {
            "city": city,
            "service_types": self.RANDOM_SERVICE_TYPES,
            "condition": condition,
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("city, geojson, expected_code", [
        (enums.CitiesEnum.SAINT_PETERSBURG, CitiesPolygonForTrafficsCalculation.SAINT_PETERSBURG_INSIDE_GEOJSON, 200),
        (enums.CitiesEnum.KRASNODAR, CitiesPolygonForTrafficsCalculation.KRASNODAR_INSIDE_GEOJSON, 400),
        (enums.CitiesEnum.SEVASTOPOL, CitiesPolygonForTrafficsCalculation.SEVASTOPOL_INSIDE_GEOJSON, 400),
    ])
    @pytest.mark.parametrize("condition", enums.ClusterizationConditionsEnum)
    def test_get_services_clusterization_with_geojson_param(self, client, city, geojson, condition, expected_code):
        """ Запрос с передачей geojson. """
        url = self.URL + "/get_clusters_polygons"
        data = {
            "city": city,
            "geojson": geojson,
            "service_types": self.RANDOM_SERVICE_TYPES,
            "condition": condition,
        }

        resp = client.post(url, json=data)
        assert resp.status_code == expected_code

    @pytest.mark.parametrize("city, area_type, area_id", [*ADMINISTRATIVE_UNITS, *MUNICIPALITIES, *BLOCKS])
    @pytest.mark.parametrize("condition", enums.ClusterizationConditionsEnum)
    def test_get_services_clusterization_area_type_and_id(self, client, city, area_type, area_id, condition):
        url = self.URL + "/get_clusters_polygons"
        data = {
            "city": city,
            "service_types": self.RANDOM_SERVICE_TYPES,
            "condition": condition,
            "area_type": area_type,
            "area_id": area_id
        }
        resp = client.post(url, json=data)
        expected_status_codes = [
            200,  # OK
            400,  # There is no services whithin a given territory
        ]
        assert resp.status_code in expected_status_codes

    @pytest.mark.parametrize("city", enums.CitiesEnum)
    @pytest.mark.parametrize("condition", enums.ClusterizationConditionsEnum)
    def test_get_services_clusterization_without_objects_to_cluster(self, client, city, condition):
        """ Тестирование случаев, когда нет сервисов для кластеризации. """
        url = self.URL + "/get_clusters_polygons"

        service_types = ["does_not_exists_service"]
        data = {
            "city": city,
            "service_types": service_types,
            "condition": condition,
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 400

        error_detail = {"detail": "There is no services whithin a given territory."}
        assert error_detail == resp.json()


class TestSpacematrix:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/spacematrix"

    @pytest.mark.parametrize("city", enums.CitiesEnum)
    def test_get_spacematrix_indices(self, client, city):
        """Запрос с обязательными полями и значениями по умолчанию. """
        data = {
            "city": city,
        }

        url = self.URL + "/get_indices"
        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("city, geojson", [
        (enums.CitiesEnum.SAINT_PETERSBURG, CitiesPolygonForTrafficsCalculation.SAINT_PETERSBURG_INSIDE_GEOJSON,),
        (enums.CitiesEnum.KRASNODAR, CitiesPolygonForTrafficsCalculation.KRASNODAR_INSIDE_GEOJSON, ),
        pytest.param(enums.CitiesEnum.SEVASTOPOL, CitiesPolygonForTrafficsCalculation.SEVASTOPOL_INSIDE_GEOJSON),
    ])
    def test_get_spacematrix_indices_geojson(self, client, city, geojson):
        """ Запрос со передачей geojson геометрии. """
        data = {
            "city": city,
            "geojson": geojson
        }

        url = self.URL + "/get_indices"
        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("city, area_type, area_id", [*ADMINISTRATIVE_UNITS, *MUNICIPALITIES, *BLOCKS])
    def test_get_spacematrix_indices_with_area_type_and_id(self, client, city, area_type, area_id):
        url = self.URL + "/get_indices"
        data = {
            "city": city,
            "area_type": area_type,
            "area_id": area_id
        }
        resp = client.post(url, json=data)
        assert resp.status_code == 200


class TestMobilityAnalysisIsochrones:
    """ Проверка метрики доступности. """
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/mobility_analysis/isochrones"
    CITIES_FROM_POINTS = [  # random points in city bbox. latitude, longitude
        (enums.CitiesEnum.SAINT_PETERSBURG, 59.9386300, 30.3141300),
        (enums.CitiesEnum.KRASNODAR, 45.0448400, 38.9760300),
        (enums.CitiesEnum.SEVASTOPOL, 44.5888300, 33.5224000)
    ]

    @pytest.mark.parametrize("travel_type", enums.MobilityAnalysisIsochronesTravelTypeEnum)
    @pytest.mark.parametrize("weight_type, weight_value", [
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.TIME, 1),
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.METER, 100)
    ])
    @pytest.mark.parametrize("city, x_from, y_from", CITIES_FROM_POINTS)
    def test_mobility_analysis_isochrones(
            self, client, city, x_from, y_from, weight_type, weight_value, travel_type
    ):
        """ Проверка вычисления изохрон для всех типов транспорта. """
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
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.TIME, 1),
    ])
    @pytest.mark.parametrize("city, x_from, y_from", CITIES_FROM_POINTS)
    def test_mobility_analysis_isochrones_is_support_routers(
            self, client, city, x_from, y_from, weight_type, weight_value, travel_type,
    ):
        """ Проверка успешного получения routers для изохрон """
        params = dict(
            city=city,
            travel_type=travel_type,
            weight_type=weight_type,
            weight_value=weight_value,
            x_from=x_from,
            y_from=y_from,
            routes=True,  # получить маршруты изохрон
        )

        url = self.URL

        resp = client.get(url, params=params)
        assert resp.status_code == 200

    @pytest.mark.parametrize("travel_type", [
        enums.MobilityAnalysisIsochronesTravelTypeEnum.WALK,
        enums.MobilityAnalysisIsochronesTravelTypeEnum.DRIVE,
    ])
    @pytest.mark.parametrize("weight_type, weight_value", [
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.TIME, 1),
        (enums.MobilityAnalysisIsochronesWeightTypeEnum.METER, 100)
    ])
    @pytest.mark.parametrize("city, x_from, y_from", CITIES_FROM_POINTS)
    def test_mobility_analysis_isochrones_is_not_support_routers(
            self, client, city, x_from, y_from, weight_type, weight_value, travel_type
    ):
        """ Проверка ошибки при получении routers для изохрон """
        params = dict(
            city=city,
            travel_type=travel_type,
            weight_type=weight_type,
            weight_value=weight_value,
            x_from=x_from,
            y_from=y_from,
            routes=True,  # получить маршруты изохрон
        )

        url = self.URL

        resp = client.get(url, params=params)
        assert resp.status_code == 422


class TestDiversity:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/diversity"
    RANDOM_SERVICE_TYPE = "universities"

    @pytest.mark.parametrize("service_type", [RANDOM_SERVICE_TYPE])
    @pytest.mark.parametrize("city", enums.CitiesEnum)
    def test_get_diversity(self, client, city, service_type):
        url = self.URL + "/diversity"
        params = {
            "city": city,
            "service_type": service_type
        }

        resp = client.get(url, params=params)
        assert resp.status_code == 200

    @pytest.mark.parametrize("service_type", ["cafes", "bakeries"])
    @pytest.mark.parametrize("city, _, block_id", BLOCKS)
    def test_get_diversity_get_buildings(self, client, city, _, block_id, service_type):
        url = self.URL + "/get_buildings"
        params = {
            "city": city,
            "block_id": block_id,
            "service_type": service_type,
        }

        resp = client.get(url, params=params)
        assert resp.status_code == 200

    @pytest.mark.parametrize("service_type", [RANDOM_SERVICE_TYPE])
    @pytest.mark.parametrize("city, house_id", [
        (enums.CitiesEnum.SAINT_PETERSBURG, 915),
        (enums.CitiesEnum.KRASNODAR, 137701),
        (enums.CitiesEnum.SEVASTOPOL, 819244),
    ])
    def test_get_diversity_get_info(self, client, city, house_id, service_type):
        url = self.URL + "/get_info"
        params = {
            "city": city,
            "house_id": house_id,
            "service_type": service_type,
        }

        resp = client.get(url, params=params)
        assert resp.status_code == 200


class TestProvision:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/provision"

    def test_get_provision(self, client):
        url = self.URL + "/get_provision"

        data = {
            "city": "saint-petersburg",
            "service_type": "kindergartens",
            "valuation_type": "normative",
            "year": 2022,
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200

    @pytest.mark.parametrize("user_changes_buildings", [
        None, provision_geojson_examples.provisions_tests_kinders_houses,
    ])
    @pytest.mark.parametrize("user_changes_services", [
        None, provision_geojson_examples.provisions_tests_kinders,
    ])
    def test_recalculate_provisions(self, client, user_changes_buildings, user_changes_services):
        url = self.URL + "/recalculate_provisions"

        data = {
            "city": "saint-petersburg",
            "service_type": "kindergartens",
            "valuation_type": "normative",
            "year": 2022,
            "user_changes_buildings": user_changes_buildings,
            "user_changes_services": user_changes_services,
            "user_provisions": provision_geojson_examples.provisions_tests_kinders_provisions,
        }

        resp = client.post(url, json=data)
        assert resp.status_code == 200


class TestCollocationMatrix:
    URL = f"http://{testing_settings.APP_ADDRESS_FOR_TESTING}/collocation_matrix"

    @pytest.mark.parametrize("city", enums.CitiesEnum)
    def test_get_collocation_matrix(self, client, city):
        """ Тестирование collocation matrix для городов. """
        url = self.URL + "/collocation_matrix"

        params = {
            "city": city,
        }

        resp = client.get(url, params=params)
        assert resp.status_code == 200
