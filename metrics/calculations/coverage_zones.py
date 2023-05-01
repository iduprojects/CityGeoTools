from typing import Optional
import json

from .errors import NormativeError
from .base_method import BaseMethod
from .mobility_analysis import AccessibilityIsochrones_v2


class CoverageZones(BaseMethod):
    """
    Coverage_Zones provides visual analytics on coverage areas of a chosen type of urban services 
    via one of given methods: (1) radius or (2) isochrone.

    '''
    Attributes
    ------
    city_model
            City Information Model

    Methods
    ------
    get_radius_zone(service_type, radius)
            Creates a buffer with a defined radius for each service.
    _get_isochrone_zone(service_type, travel_type, weight_value)
            Creates an isochrone with defined way of transportation and time to travel (in minutes) for each service. 
    """

    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("coverage_zones")
        self.service_types = self.city_model.ServiceTypes.copy()
        self.services = self.city_model.Services.copy()
        self.walk_speed = 4 * 1000 / 60

    def get_radius_zone(self, service_type: str, radius: Optional[int]):
        """
        Creates a buffer with a defined radius for each service.

        Parameters
        ---------
        service_type: str
            The type of services to run the method on.
        radius: int, optional
            The radius for the buffer.
            If radius argument is not defined, it tries to get the value from ServicesTypes's standards.
        
        Returns
        -------
        FeatureCollection

        Errors
        ------
        Raises NormativeError if radius (with given radius=None) cannot be defined from ServiceTypes.

        Example
        -------
        Get coverage zones for schools with radius of 50 meters.
            CityMetricsMethods.Coverage_Zones(city_model).get_radius_zone(service_type='schools', radius=50)
        """

        service_types  = self.service_types
        services = self.services[self.services['service_code'] == service_type].reset_index(drop=True)

        if not radius:
            if service_types[service_types['code'] == service_type]['walking_radius_normative'].notna().iloc[0]:
                    radius = service_types[service_types['code'] == service_type].iloc[0]['walking_radius_normative']
            elif service_types[service_types['code'] == service_type]['public_transport_time_normative'].notna().iloc[0]:
                    radius = service_types[service_types['code'] == service_type]
                    radius = radius.iloc[0]['public_transport_time_normative'] * self.walk_speed
            else:
                raise NormativeError("radius", service_type)
        
        
        services['geometry'] = services['geometry'].buffer(radius)

        return json.loads(services.reset_index().to_crs(4326).to_json())


    def get_isochrone_zone(self, service_type: str, travel_type:str, weight_value: int):
        """
        Creates an isochrone with defined way of transportation and time to travel (in minutes) for each service.
        The method calls Accessibility_Isochrones_v2.get_isochrone.

        Parameters
        ---------
        service_type: str
            The type of services to run the method on.
        travel_type: str
            From Accessibility_Isochrones_v2. 
            One of the given ways of transportation: "public_transport", "walk" or "drive".
        weight_value: int
            From Accessibility_Isochrones_v2.
            Minutes to travel.
        
        Returns
        -------
        FeatureCollection

        Example
        -------
        Get coverage zones for dentistries using 10 mins pedestrian-ways isochrone.
            CityMetricsMethods.Coverage_Zones(city_model)._get_isochrone_zone(
                service_type='dentists', travel_type='walk', weight_value = 10)
        """

        services = self.services[self.services['service_code'] == service_type].reset_index(drop=True)

        x_from = services['geometry'].x
        y_from = services['geometry'].y
        
        isochrone = AccessibilityIsochrones_v2(self.city_model).get_isochrone(
            travel_type, x_from, y_from, weight_value, weight_type = 'time_min')
        
        return isochrone["isochrone"]