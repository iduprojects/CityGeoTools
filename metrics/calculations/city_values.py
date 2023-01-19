
import pandas as pd
import numpy as np
import geopandas as gpd
import json

from typing import Any, Optional
from .base_method import BaseMethod
from .city_provision import CityProvision


class CityValues(BaseMethod):
    def __init__(self, city_model: Any,
                 valuation_type: str, year: int, ):

        self.city_model = city_model
        self.valuation_type = valuation_type
        self.year = year
        self.ServiceTypes = city_model.ServiceTypes.copy(deep = True)
        self.ServiceTypes.index = self.ServiceTypes.id
        self.city_model.SocialGroups = self.city_model.SocialGroups.copy(deep = True)
        self.city_model.SocialGroups.index = self.city_model.SocialGroups.social_groups_id
        self.city_model.ValueTypes = self.city_model.ValueTypes.copy(deep = True)
        self.city_model.ValueTypes.index = self.city_model.ValueTypes.value_type_id

        grouped = []
        for value_type_id in pd.unique(self.city_model.SocialGroupsValueTypesLivingSituations.value_type_id): 
            t = self.city_model.SocialGroupsValueTypesLivingSituations[self.city_model.SocialGroupsValueTypesLivingSituations['value_type_id'] == value_type_id].groupby(by = 'social_group_id')['living_situation_id'].apply(list)
            t.name = value_type_id
            grouped.append(t)

        self.values_provision = pd.DataFrame(grouped)
        self.values_provision = self.values_provision.apply(lambda x: self._transform_living_sit_to_service_type(x), axis = 1)
        services_list_values_provision = [list(set(x[1].dropna().sum())) for x in self.values_provision.iterrows()]
        services_list_values_provision = set([item for sublist in services_list_values_provision for item in sublist])
        self.ServiceTypes = self.ServiceTypes.loc[services_list_values_provision]
        self.ServiceTypes.index = self.ServiceTypes['code']
        self.ServiceTypes['city_provision_value'] = self.ServiceTypes.apply(lambda x: self._get_city_provissions_value(x), axis = 1)
        self.ServiceTypes.index = self.ServiceTypes.id

        self.values_provision = self.values_provision.apply(lambda x: self._assign_provisions_to_values(x.dropna()))
        self.values_provision.columns = self.city_model.SocialGroups.loc[self.values_provision.columns]['social_groups_name']
        self.values_provision.index = self.city_model.ValueTypes.loc[self.values_provision.index]['value_types_name']

    def _get_service_types(self, l_s_selection):
            try:
                return list(self.city_model.LivingSituationsCityServiceTypes.loc[l_s_selection]['city_service_type_id'].unique())
            except:
                return np.NAN

    def _transform_living_sit_to_service_type(self, loc):
            loc = loc.dropna().apply(lambda x: self._get_service_types(x))
            return loc

    def _get_city_provissions_value (self, s_t_loc):
        r = CityProvision(city_model = self.city_model, 
                             service_types = [s_t_loc.name],
                             valuation_type = self.valuation_type, 
                             year = self.year,
                             service_impotancy = [1],
                             return_jsons = False,
                             calculation_type = 'gravity').get_provisions()

        return r.Provisions[s_t_loc.name]['buildings'][f'{s_t_loc.name}_provison_value'].mean() 

    def _assign_provisions_to_values(self, loc):
        return loc.apply(lambda x: self.ServiceTypes.loc[x]['city_provision_value'].mean().round(2))

    def get_city_values(self, ):


        return {"city_values":json.loads(self.values_provision.to_json())}