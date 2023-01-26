
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
        self.valuation_type = valuation_type

        self.ServiceTypes = city_model.ServiceTypes.copy(deep = True)
        self.ServiceTypes.index = self.ServiceTypes.id
        self.SocialGroups = self.city_model.SocialGroups.copy(deep = True)
        self.SocialGroups.index = city_model.SocialGroups.social_groups_id
        self.ValueTypes = self.city_model.ValueTypes.copy(deep = True)
        self.LivingSituations = self.city_model.LivingSituations.copy(deep = True)
        
        self.LivingSituationsCityServiceTypes = self.city_model.LivingSituationsCityServiceTypes.copy(deep = True)

        self.SocialGroupsValueTypesLivingSituations = self.city_model.SocialGroupsValueTypesLivingSituations.copy(deep = True)

        t = self.city_model.LivingSituationsCityServiceTypes.groupby('living_situation_id')['city_service_type_id'].apply(list)
        self.SocialGroupsValueTypesLivingSituations = self.SocialGroupsValueTypesLivingSituations.merge(t, 
                                                                                                        left_on = 'living_situation_id', 
                                                                                                        right_index=True, 
                                                                                                        how = 'left')
        self.SocialGroupsValueTypesLivingSituations = self.SocialGroupsValueTypesLivingSituations.merge(self.ValueTypes[['value_type_id','value_group_id']], 
                                                                                                        left_on = 'value_type_id', 
                                                                                                        right_on = 'value_type_id', 
                                                                                                        how = 'left')
        self.SocialGroupsValueTypesLivingSituations = self.SocialGroupsValueTypesLivingSituations.merge(self.LivingSituations,
                                                                                                        left_on = 'living_situation_id', 
                                                                                                        right_on = 'living_situation_id', 
                                                                                                        how = 'left')

        services_unique_list = set(self.SocialGroupsValueTypesLivingSituations['city_service_type_id'].dropna().sum())
        self.ServiceTypes = self.ServiceTypes.loc[services_unique_list]
        self.ServiceTypes.index = self.ServiceTypes['code']
        self.ServiceTypes['city_provision_value'] = self.ServiceTypes.apply(lambda x: self._get_city_provissions_value(x), axis = 1)
        self.ServiceTypes['city_provision_value'] = self.ServiceTypes['city_provision_value'].round(2) 

        index = self.SocialGroupsValueTypesLivingSituations[['value_group_id','value_type_id']].drop_duplicates()
        index = pd.MultiIndex.from_tuples([list(a) for a in index.values], names = ['value_group_id','value_type_id'])
        self.city_values = pd.DataFrame(np.NaN,index = index, columns = self.SocialGroupsValueTypesLivingSituations['social_group_id'].drop_duplicates())
        self.ServiceTypes.index = self.ServiceTypes.id.values
        self.city_values = self.city_values.apply(lambda x: self._assign_provisions_to_values(x), axis = 1)
        self.city_values.columns = self.SocialGroups.loc[self.city_values.columns]['social_groups_name']
        self.ValueTypes.index = self.ValueTypes['value_group_id']
        self.city_values.index = self.city_values.index.set_levels(self.ValueTypes.loc[self.city_values.index.levels[0].values]['value_group'].drop_duplicates().values, level=0) 
        self.ValueTypes.index = self.ValueTypes['value_type_id']
        self.city_values.index = self.city_values.index.set_levels(self.ValueTypes.loc[self.city_values.index.levels[1].values]['value_type'].drop_duplicates().values, level=1)
    
    
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
        value_type_id = loc.name[1]
        results = []
        for social_group_id in loc.index:
            s = self.SocialGroupsValueTypesLivingSituations[(self.SocialGroupsValueTypesLivingSituations['value_type_id'] == value_type_id) & (self.SocialGroupsValueTypesLivingSituations['social_group_id'] == social_group_id)]
            s = s[['living_situations_name','city_service_type_id']].dropna()
            if len(s) > 0:
                t = dict(zip(s['living_situations_name'].values, s['city_service_type_id'].values))
                for k, v in t.items():
                    a = self.ServiceTypes.loc[v]
                    t[k] = dict(zip(a['name'].values, a['city_provision_value'].values))
                    t[k]['situation_mean'] = a['city_provision_value'].mean().round(2)
                t['value_mean'] = np.mean([v['situation_mean'] for v in t.values()]).round(2)
                results.append(t)
            else:
                results.append(np.NAN)
        return pd.Series(data = results, index = loc.index)     

    def get_city_values(self, ):
        _json = {}
        for value_group_id, part in self.city_values.groupby(level=0):
            _json[value_group_id] = eval(part.droplevel(level = 0).to_json(orient = 'index').replace('null', 'None'))
        return {'city_values': _json}