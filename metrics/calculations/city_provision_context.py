import requests
import os
import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
import shapely.wkt
import io
import networkit as nk
import pulp
from multiprocesspandas import applyparallel

from typing import Any, Optional
from .city_provision import CityProvision

class CityProvisionContext(CityProvision): 
    def __init__(self, city_model: Any, service_types: list, 
                 valuation_type: str, year: int, 
                 user_context_zone: Optional[dict]):
        
        super().__init__(city_model=city_model, service_types=service_types, 
                                           valuation_type=valuation_type, year=year,
                                           user_provisions=None, user_changes_buildings=None, 
                                           user_changes_services=None,user_selection_zone=None, service_impotancy=None,
                                           return_jsons = False
                                           )

        self.AdministrativeUnits = city_model.AdministrativeUnits.copy(deep = True) 
            
        self.get_provisions()
        if user_context_zone:
            gdf = gpd.GeoDataFrame(data = {"id":[1]}, 
                                    geometry = [shapely.geometry.shape(user_context_zone)],
                                    crs = city_model.city_crs)
            self.user_context_zone = gdf['geometry'][0]
        else:
            self.user_context_zone = None
    @staticmethod
    def _extras(buildings, services, extras, service_types):
        extras['top_10_services'] = {s_t: eval(services.get_group(s_t).sort_values(by = 'service_load').tail(10).to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')) for s_t in service_types}
        extras['bottom_10_services'] = {s_t: eval(services.get_group(s_t).sort_values(by = 'service_load').head(10).to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')) for s_t in service_types}
        extras['top_10_houses'] = {s_t: eval(buildings.sort_values(by = s_t + '_provison_value').tail(10).to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')) for s_t in service_types}
        extras['bottom_10_houses'] = {s_t: eval(buildings.sort_values(by = s_t + '_provison_value').head(10).to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')) for s_t in service_types}
        extras['top_10_houses_total'] = eval(buildings.sort_values(by = 'total_provision_assessment').tail(10).to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False'))
        extras['bottom_10_houses_total'] = eval(buildings.sort_values(by = 'total_provision_assessment').head(10).to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False'))
        
        return extras 


    def get_context(self, ):
        #provisions values total and individual
        selection_cols_means = [s_t + '_provison_value' for s_t in self.service_types] + ['total_provision_assessment']
        #total individual services demand in area 
        #unsupplyed demand
        #supplyed demands within
        #supplyed demands without
        selection_cols_sums = [s_t+'_service_demand_value_normative' for s_t in self.service_types] \
        + [s_t+'_service_demand_left_value_normative' for s_t in self.service_types] \
        + [s_t+'_supplyed_demands_within' for s_t in self.service_types] \
        + [s_t+'_supplyed_demands_without' for s_t in self.service_types]    
        extras = {}
        if self.user_context_zone:
            a = self.buildings.within(self.user_context_zone)
            selection_buildings = self.buildings.loc[a[a].index]
            a = self.services.within(self.user_context_zone)
            selection_services = self.services.loc[a[a].index]
            services_grouped = selection_services.groupby(by = ['service_code'])

            services_self_data = pd.concat([services_grouped.sum().loc[s_t][['capacity','capacity_left']].rename({'capacity':s_t + '_capacity', 
                                                                                                                  'capacity_left':s_t + '_capacity_left'}) for s_t in self.service_types])
            self.zone_context = gpd.GeoDataFrame(data = [pd.concat([selection_buildings.mean()[selection_cols_means],
                                                                    selection_buildings.sum()[selection_cols_sums],
                                                                    services_self_data])], 
                                                 geometry = [self.user_context_zone], 
                                                 crs = self.user_context_zone.crs)
            extras = self._extras(selection_buildings, services_grouped, extras, self.service_types)
            self.zone_context = self.zone_context.to_crs(4326)
            self.zone_context = self.zone_context.drop(columns = [x for x in self.zone_context.columns if x.split('_')[0] in self.service_types if not '_provison_value' in x])
            return {"context_unit": eval(self.zone_context.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')),
                    "additional_data": extras}
        else:
            grouped_buildings = self.buildings.groupby(by = 'administrative_unit_id')
            services_grouped = self.services.groupby(by = ['service_code','administrative_unit_id'])
            grouped_buildings_means = grouped_buildings.mean()
            grouped_buildings_sums = grouped_buildings.sum()
            self.AdministrativeUnits = self.AdministrativeUnits.merge(pd.merge(grouped_buildings_means[selection_cols_means], 
                                                                               grouped_buildings_sums[selection_cols_sums], 
                                                                               left_index = True, 
                                                                               right_index = True), left_on = 'id', right_index = True)
            #services original capacity and left capacity 
            services_context_data = pd.concat([services_grouped.sum().loc[s_t][['capacity','capacity_left']].rename(columns = {'capacity':s_t + '_capacity', 
                                                                                                                               'capacity_left':s_t + '_capacity_left'}) for s_t in self.service_types], axis = 1)
            self.AdministrativeUnits = self.AdministrativeUnits.merge(services_context_data, left_on = 'id', right_index = True)
            self.AdministrativeUnits = self.AdministrativeUnits.fillna(0)

            services_grouped = self.services.groupby(by = ['service_code'])
            extras = self._extras(self.buildings, services_grouped, extras, self.service_types)
            self.AdministrativeUnits = self.AdministrativeUnits.to_crs(4326)
            self.AdministrativeUnits = self.AdministrativeUnits.drop(columns = [x for x in self.AdministrativeUnits.columns if x.split('_')[0] in self.service_types if not '_provison_value' in x])
            return {"context_unit": eval(self.AdministrativeUnits.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')),
                    "additional_data": extras}


