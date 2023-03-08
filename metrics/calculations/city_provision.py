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

from typing import Any, Optional
from .base_method import BaseMethod

class CityProvision(BaseMethod): 

    def __init__(self, city_model: Any, service_types: list, valuation_type: str, year: int,
                 user_provisions: Optional[dict[str, dict]] = None, 
                 user_changes_buildings: Optional[dict] = None,
                 user_changes_services: Optional[dict] = None,
                 user_selection_zone: Optional[dict] = None,
                 service_impotancy: Optional[list] = None,
                 return_jsons: bool = False,
                 calculation_type:str = 'gravity'
                 ):
        '''
        >>> 
        >>>
        >>>
        '''
        BaseMethod.__init__(self, city_model)
        self.engine = city_model.engine
        self.city_name = city_model.city_name
        self.service_types = service_types
        self.valuation_type = valuation_type
        self.year = year
        self.city = city_model.city_name
        self.service_types_normatives = city_model.ServiceTypes[city_model.ServiceTypes['code'].isin(service_types)].copy(deep = True)
        self.service_types_normatives.index = self.service_types_normatives['code'].values
        self.return_jsons = return_jsons
        self.graph_nk_length = city_model.graph_nk_length
        self.graph_nk_time =  city_model.graph_nk_time
        self.nx_graph =  city_model.MobilityGraph
        self.buildings = city_model.Buildings.copy(deep = True)
        self.buildings = self.buildings.dropna(subset = 'functional_object_id')
        self.buildings['functional_object_id'] = self.buildings['functional_object_id'].astype(int)
        self.buildings.index = self.buildings['functional_object_id'].values
        self.calculation_type = calculation_type
        
        self.services = city_model.Services[city_model.Services['service_code'].isin(service_types)].copy(deep = True)
        self.services.index = self.services['id'].values.astype(int)

        self.file_server = "http://10.32.1.60:8090/"
        #self.file_server = os.environ['PROVISIONS_DATA_FILE_SERVER']

        try:
            self.services_impotancy = dict(zip(service_types, service_impotancy))
        except:
            self.services_impotancy = None
        self.user_provisions = {}
        self.user_changes_buildings = {}
        self.user_changes_services = {}
        self.buildings_old_values = None
        self.services_old_values = None
        self.errors = []
        #try:
        self.demands = pd.read_sql(f'''SELECT functional_object_id, {", ".join(f"{service_type}_service_demand_value_{self.valuation_type}" for service_type in service_types)}
                                FROM social_stats.buildings_load_future
                                WHERE year = {self.year}
                                ''', con = self.engine)
        self.demands.index = self.demands['functional_object_id'].values
        self.buildings = self.buildings.merge(self.demands, left_index = True, right_on = ['functional_object_id'], how = 'left')

        #except:
            #self.errors.append(service_type)
        for service_type in service_types:
            self.buildings[f'{service_type}_service_demand_left_value_{self.valuation_type}'] = self.buildings[f'{service_type}_service_demand_value_{self.valuation_type}']
            self.buildings = self.buildings.dropna(subset = f'{service_type}_service_demand_value_{self.valuation_type}')
            
        self.service_types= [x for x in service_types if x not in self.errors]
        self.buildings.index = self.buildings['functional_object_id'].values.astype(int)
        self.services['capacity_left'] = self.services['capacity']
        self.Provisions = {service_type:{'destination_matrix': None, 
                                         'distance_matrix': None,
                                         'normative_distance':None,
                                         'buildings':None,
                                         'services': None,
                                         'selected_graph':None} for service_type in service_types}
        self.new_Provisions = {service_type:{'destination_matrix': None, 
                                            'distance_matrix': None,
                                            'normative_distance':None,
                                            'buildings':None,
                                            'services': None,
                                            'selected_graph':None} for service_type in service_types}
        #Bad interface , raise error must be 
        if user_changes_services:
            self.user_changes_services = gpd.GeoDataFrame.from_features(user_changes_services['features']).set_crs(4326).to_crs(self.city_crs)
            self.user_changes_services.index = self.user_changes_services['id'].values.astype(int)
            self.user_changes_services = self.user_changes_services.combine_first(self.services)
            self.user_changes_services.index = self.user_changes_services['id'].values.astype(int)
            self.user_changes_services['capacity_left'] = self.user_changes_services['capacity']
            self.services_old_values = self.user_changes_services[['capacity','capacity_left','carried_capacity_within','carried_capacity_without']]
            self.user_changes_services = self.user_changes_services.set_crs(self.city_crs)
            self.user_changes_services.index = self.user_changes_services['id'].values.astype(int)
            self.user_changes_services.geometry = self.user_changes_services.centroid
        else:
            self.user_changes_services = self.services.copy(deep = True)
        if user_changes_buildings:
            old_cols = []
            self.user_changes_buildings = gpd.GeoDataFrame.from_features(user_changes_buildings['features']).set_crs(4326).to_crs(self.city_crs)
            self.user_changes_buildings.index = self.user_changes_buildings['functional_object_id'].values.astype(int)
            self.user_changes_buildings = self.user_changes_buildings.combine_first(self.buildings)
            self.user_changes_buildings.index = self.user_changes_buildings['functional_object_id'].values.astype(int)
            for service_type in service_types:
                old_cols.extend([f'{service_type}_provison_value', 
                                 f'{service_type}_service_demand_left_value_{self.valuation_type}', 
                                 f'{service_type}_service_demand_value_{self.valuation_type}', 
                                 f'{service_type}_supplyed_demands_within', 
                                 f'{service_type}_supplyed_demands_without'])
                self.user_changes_buildings[f'{service_type}_service_demand_left_value_{self.valuation_type}'] = self.user_changes_buildings[f'{service_type}_service_demand_value_{self.valuation_type}'].values
            self.buildings_old_values = self.user_changes_buildings[old_cols]
            self.user_changes_buildings = self.user_changes_buildings.set_crs(self.city_crs)
            self.user_changes_buildings.index = self.user_changes_buildings['functional_object_id'].values.astype(int)
        else:
            self.user_changes_buildings = self.buildings.copy()
        if user_provisions:
            for service_type in service_types:
                self.user_provisions[service_type] = gpd.GeoDataFrame.from_features(user_provisions[service_type]['features']).set_crs(4326).to_crs(self.city_crs)
        else:
            self.user_provisions = None
        if user_selection_zone:
            gdf = gpd.GeoDataFrame(data = {"id":[1]}, 
                                    geometry = [shapely.geometry.shape(user_selection_zone)],
                                    crs = 4326).to_crs(city_model.city_crs)
            self.user_selection_zone = gdf['geometry'][0]
        else:
            self.user_selection_zone = None

    def get_provisions(self, ):
        
        for service_type in self.service_types:
            normative_distance = self.service_types_normatives.loc[service_type].dropna().copy(deep = True)
            try:
                self.Provisions[service_type]['normative_distance'] = normative_distance['walking_radius_normative']
                self.Provisions[service_type]['selected_graph'] = self.graph_nk_length
            except:
                self.Provisions[service_type]['normative_distance'] = normative_distance['public_transport_time_normative']
                self.Provisions[service_type]['selected_graph'] = self.graph_nk_time
            
            try:
                self.Provisions[service_type]['services'] = pd.read_pickle(io.BytesIO(requests.get(f'{self.file_server}provision_1/{self.city_name}_{service_type}_{self.year}_{self.valuation_type}_services').content))
                self.Provisions[service_type]['buildings'] = pd.read_pickle(io.BytesIO(requests.get(f'{self.file_server}provision_1/{self.city_name}_{service_type}_{self.year}_{self.valuation_type}_buildings').content))
                self.Provisions[service_type]['distance_matrix'] = pd.read_pickle(io.BytesIO(requests.get(f'{self.file_server}provision_1/{self.city_name}_{service_type}_{self.year}_{self.valuation_type}_distance_matrix').content))
                self.Provisions[service_type]['destination_matrix'] = pd.read_pickle(io.BytesIO(requests.get(f'{self.file_server}provision_1/{self.city_name}_{service_type}_{self.year}_{self.valuation_type}_destination_matrix').content))
                print(service_type + ' loaded')
            except:
                print(service_type + ' not loaded')
                self.Provisions[service_type]['buildings'] = self.buildings.copy(deep = True)
                self.Provisions[service_type]['services'] = self.services[self.services['service_code'] == service_type].copy(deep = True)    
                self.Provisions[service_type] =  self._calculate_provisions(self.Provisions[service_type], service_type, calculation_type = self.calculation_type)
                self.Provisions[service_type]['buildings'], self.Provisions[service_type]['services'] = self._additional_options(self.Provisions[service_type]['buildings'].copy(), 
                                                                                                                                    self.Provisions[service_type]['services'].copy(),
                                                                                                                                        self.Provisions[service_type]['distance_matrix'].copy(),
                                                                                                                                        self.Provisions[service_type]['destination_matrix'].copy(),
                                                                                                                                        self.Provisions[service_type]['normative_distance'],
                                                                                                                                        service_type,
                                                                                                                                        self.user_selection_zone,
                                                                                                                                        self.valuation_type)
        cols_to_drop = [x for x in self.buildings.columns for service_type in self.service_types if service_type in x]
        self.buildings = self.buildings.drop(columns = cols_to_drop)
        for service_type in self.service_types: 
            self.buildings = self.buildings.merge(self.Provisions[service_type]['buildings'], 
                                                    left_on = 'functional_object_id', 
                                                    right_on = 'functional_object_id')
        to_rename_x = [x for x in self.buildings.columns if '_x' in x]
        to_rename_y = [x for x in self.buildings.columns if '_y' in x]
        self.buildings = self.buildings.rename(columns = dict(zip(to_rename_x, [x.split('_x')[0] for x in to_rename_x])))
        self.buildings = self.buildings.rename(columns = dict(zip(to_rename_y, [y.split('_y')[0] for y in to_rename_y])))
        self.buildings = self.buildings.loc[:,~self.buildings.columns.duplicated()].copy()
        self.buildings.index = self.buildings['functional_object_id'].values.astype(int)
        self.services = pd.concat([self.Provisions[service_type]['services'] for service_type in self.service_types])
        self.buildings, self.services = self._is_shown(self.buildings,self.services, self.Provisions)
        self.buildings = self._provisions_impotancy(self.buildings)
        self.buildings = self.buildings.fillna(0)
        self.services = self.services.fillna(0)
        self.services = self.services.to_crs(4326)
        self.buildings = self.buildings.to_crs(4326)
        if self.return_jsons == True:  
            return {"houses": eval(self.buildings.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')), 
                    "services": eval(self.services.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')), 
                    "provisions": {service_type: eval(self._provision_matrix_transform(self.Provisions[service_type]['destination_matrix'], 
                                                                                       self.services[self.services['is_shown'] == True],
                                                                                       self.buildings[self.buildings['is_shown'] == True]).to_json().replace('null', 'None')) for service_type in self.service_types}}
        else:
            return self

    def _provisions_impotancy(self, buildings):
        provision_value_columns = [service_type + '_provison_value' for service_type in self.service_types]
        if self.services_impotancy:
            t = buildings[provision_value_columns].apply(lambda x: self.services_impotancy[x.name.split("_provison_value")[0]]*x).sum(axis = 1)
        else: 
            t = buildings[provision_value_columns].sum(axis = 1)
        _min = t.min()
        _max = t.max()
        t = t.apply(lambda x: (x - _min)/(_max - _min))
        buildings['total_provision_assessment'] = t
        return buildings

    def _is_shown(self, buildings, services, Provisions):
        if self.user_selection_zone:
            buildings['is_shown'] = buildings.within(self.user_selection_zone)
            a = buildings['is_shown'].copy() 
            t = []
            for service_type in self.service_types:
                t.append(Provisions[service_type]['destination_matrix'][a[a].index.values].apply(lambda x: len(x[x > 0])>0, axis = 1))
            services['is_shown'] = pd.concat([a[a] for a in t])
        else:
            buildings['is_shown'] = True
            services['is_shown'] = True
        return buildings, services

    def _calculate_provisions(self, Provisions, service_type, calculation_type):
        df = pd.DataFrame.from_dict(dict(self.nx_graph.nodes(data=True)), orient='index')
        self.graph_gdf = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = self.city_crs)
        from_houses = self.graph_gdf['geometry'].sindex.nearest(Provisions['buildings']['geometry'], 
                                                                return_distance = True, 
                                                                return_all = False) 
        to_services = self.graph_gdf['geometry'].sindex.nearest(Provisions['services']['geometry'], 
                                                                return_distance = True, 
                                                                return_all = False)
        Provisions['distance_matrix'] = pd.DataFrame(0, index = to_services[0][1], 
                                                        columns = from_houses[0][1])

        splited_matrix = np.array_split(Provisions['distance_matrix'].copy(deep = True), int(len(Provisions['distance_matrix']) / 1000) + 1)
        
        for i in range(len(splited_matrix)):
            r = nk.distance.SPSP(G = Provisions['selected_graph'], 
                                sources = splited_matrix[i].index.values).run()
            splited_matrix[i] = splited_matrix[i].apply(lambda x: self._get_nk_distances(r,x), axis =1)
            del r
        Provisions['distance_matrix'] = pd.concat(splited_matrix)
        del splited_matrix

        Provisions['distance_matrix'].index = Provisions['services'].index
        Provisions['distance_matrix'].columns = Provisions['buildings'].index
        Provisions['destination_matrix'] = pd.DataFrame(0, index = Provisions['distance_matrix'].index, 
                                                           columns = Provisions['distance_matrix'].columns)
        print(Provisions['buildings'][f'{service_type}_service_demand_left_value_{self.valuation_type}'].sum(), 
              Provisions['services']['capacity_left'].sum(), 
              Provisions['normative_distance'])

        if calculation_type == 'gravity':
            Provisions['destination_matrix'] = self._provision_loop_gravity(Provisions['buildings'].copy(), 
                                                        Provisions['services'].copy(), 
                                                        Provisions['distance_matrix'].copy() + 1, 
                                                        Provisions['normative_distance'], 
                                                        Provisions['destination_matrix'].copy(),
                                                        service_type )

        elif calculation_type == 'linear':                                                            
            Provisions['destination_matrix'] = self._provision_loop_linear(Provisions['buildings'].copy(), 
                                                                    Provisions['services'].copy(), 
                                                                    Provisions['distance_matrix'].copy(), 
                                                                    Provisions['normative_distance'], 
                                                                    Provisions['destination_matrix'].copy(),
                                                                    service_type )
        return Provisions        

    @staticmethod
    def _restore_user_provisions(user_provisions):
        restored_user_provisions = user_provisions[['service_id','house_id','demand']].groupby(['service_id','house_id']).first().unstack().droplevel(level = 0, axis = 1).fillna(0)
        restored_user_provisions.index.name = None
        restored_user_provisions.columns.name = None

        return restored_user_provisions

    @staticmethod
    def _additional_options(buildings, services, Matrix, destination_matrix, normative_distance, service_type, selection_zone, valuation_type): 
        #clear matrix same size as buildings and services if user sent sth new
        cols_to_drop = list(set(set(Matrix.columns.values) - set(buildings.index.values)))
        rows_to_drop = list(set(set(Matrix.index.values) - set(services.index.values)))
        Matrix = Matrix.drop(index=rows_to_drop, 
                                columns=cols_to_drop, 
                                errors = 'irgonre')
        destination_matrix = destination_matrix.drop(index=rows_to_drop, 
                                    columns=cols_to_drop, 
                                    errors = 'irgonre')                             
        #bad performance 
        #bad code
        #rewrite to vector operations [for col in ****]
        buildings[f'{service_type}_service_demand_left_value_{valuation_type}'] = buildings[f'{service_type}_service_demand_value_{valuation_type}'] 
        buildings[f'{service_type}_supplyed_demands_within'] = 0
        buildings[f'{service_type}_supplyed_demands_without'] = 0
        services['capacity_left'] = services['capacity']
        services['carried_capacity_within'] = 0
        services['carried_capacity_without'] = 0
        for i in range(len(destination_matrix)):
            loc = destination_matrix.iloc[i]
            s = Matrix.loc[loc.name] <= normative_distance
            within = loc[s]
            without = loc[~s]
            within = within[within > 0]
            without = without[without > 0]
            buildings[f'{service_type}_service_demand_left_value_{valuation_type}'] = buildings[f'{service_type}_service_demand_left_value_{valuation_type}'].sub(within.add(without, fill_value= 0), fill_value = 0)
            buildings[f'{service_type}_supplyed_demands_within'] = buildings[f'{service_type}_supplyed_demands_within'].add(within, fill_value = 0)
            buildings[f'{service_type}_supplyed_demands_without'] = buildings[f'{service_type}_supplyed_demands_without'].add(without, fill_value = 0)
            services.at[loc.name,'capacity_left'] = services.at[loc.name,'capacity_left'] - within.add(without, fill_value= 0).sum()
            services.at[loc.name,'carried_capacity_within'] = services.at[loc.name,'carried_capacity_within'] + within.sum()
            services.at[loc.name,'carried_capacity_without'] = services.at[loc.name,'carried_capacity_without'] + without.sum()
        buildings[f'{service_type}_provison_value'] = buildings[f'{service_type}_supplyed_demands_within'] / buildings[f'{service_type}_service_demand_value_{valuation_type}']
        services['service_load'] = services['capacity'] - services['capacity_left']

        buildings = buildings[[x for x in buildings.columns if service_type in x] + ['functional_object_id']]
        return buildings, services 

    def recalculate_provisions(self, ):
        
        for service_type in self.service_types:
            print(service_type)
            normative_distance = self.service_types_normatives.loc[service_type].dropna().copy(deep = True)
            try:
                self.new_Provisions[service_type]['normative_distance'] = normative_distance['walking_radius_normative']
                self.new_Provisions[service_type]['selected_graph'] = self.graph_nk_length
                print('walking_radius_normative')
            except:
                self.new_Provisions[service_type]['normative_distance'] = normative_distance['public_transport_time_normative']
                self.new_Provisions[service_type]['selected_graph'] = self.graph_nk_time
                print('public_transport_time_normative')
            
            self.new_Provisions[service_type]['buildings'] = self.user_changes_buildings.copy(deep = True)
            self.new_Provisions[service_type]['services'] = self.user_changes_services[self.user_changes_services['service_code'] == service_type].copy(deep = True)

            self.new_Provisions[service_type] =  self._calculate_provisions(self.new_Provisions[service_type], 
                                                                            service_type, 
                                                                            calculation_type=self.calculation_type)
            self.new_Provisions[service_type]['buildings'], self.new_Provisions[service_type]['services'] = self._additional_options(self.new_Provisions[service_type]['buildings'].copy(), 
                                                                                                                                     self.new_Provisions[service_type]['services'].copy(),
                                                                                                                                     self.new_Provisions[service_type]['distance_matrix'].copy(),
                                                                                                                                     self.new_Provisions[service_type]['destination_matrix'].copy(),
                                                                                                                                     self.new_Provisions[service_type]['normative_distance'],
                                                                                                                                     service_type,
                                                                                                                                     self.user_selection_zone,
                                                                                                                                     self.valuation_type)
            self.new_Provisions[service_type]['buildings'], self.new_Provisions[service_type]['services'] = self._get_provisions_delta(service_type)
        cols_to_drop = [x for x in self.user_changes_buildings.columns for service_type in self.service_types if service_type in x]
        self.user_changes_buildings = self.user_changes_buildings.drop(columns = cols_to_drop)
        for service_type in self.service_types:
            self.user_changes_buildings = self.user_changes_buildings.merge(self.new_Provisions[service_type]['buildings'], 
                                                                            left_on = 'functional_object_id', 
                                                                            right_on = 'functional_object_id')                                                             
        to_rename_x = [x for x in self.user_changes_buildings.columns if '_x' in x]
        to_rename_y = [x for x in self.user_changes_buildings.columns if '_y' in x]
        self.user_changes_buildings = self.user_changes_buildings.rename(columns = dict(zip(to_rename_x, [x.split('_x')[0] for x in to_rename_x])))
        self.user_changes_buildings = self.user_changes_buildings.rename(columns = dict(zip(to_rename_y, [y.split('_y')[0] for y in to_rename_y])))
        self.user_changes_buildings = self.user_changes_buildings.loc[:,~self.user_changes_buildings.columns.duplicated()].copy()

        self.user_changes_buildings.index = self.user_changes_buildings['functional_object_id'].values.astype(int)
        self.user_changes_services = pd.concat([self.new_Provisions[service_type]['services'] for service_type in self.service_types])
        self.user_changes_buildings, self.user_changes_services = self._is_shown(self.user_changes_buildings,self.user_changes_services, self.new_Provisions)
        self.user_changes_buildings = self._provisions_impotancy(self.user_changes_buildings)
        self.user_changes_services = self.user_changes_services.fillna(0)
        self.user_changes_buildings = self.user_changes_buildings.fillna(0)
        self.user_changes_services = self.user_changes_services.to_crs(4326)
        self.user_changes_buildings = self.user_changes_buildings.to_crs(4326)

        if self.return_jsons == True:  
            return {"houses": eval(self.user_changes_buildings.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')), 
                    "services": eval(self.user_changes_services.to_json().replace('true', 'True').replace('null', 'None').replace('false', 'False')), 
                    "provisions": {service_type: eval(self._provision_matrix_transform(self.new_Provisions[service_type]['destination_matrix'], 
                                                                                       self.user_changes_services[self.user_changes_services['is_shown'] == True],
                                                                                       self.user_changes_buildings[self.user_changes_buildings['is_shown'] == True]).to_json().replace('null', 'None')) for service_type in self.service_types}}
        else:
            return self

    def _get_provisions_delta(self, service_type):
        #bad performance 
        #bad code
        #rewrite to df[[for col.split()[0] in ***]].sub(other[col])
        services_delta_cols = ['capacity_delta', 'capacity_left_delta', 'carried_capacity_within_delta', 'carried_capacity_without_delta']
        buildsing_delta_cols = [f'{service_type}_provison_value_delta', 
                                f'{service_type}_service_demand_left_value_{self.valuation_type}_delta', 
                                f'{service_type}_service_demand_value_{self.valuation_type}_delta',
                                f'{service_type}_supplyed_demands_within_delta',
                                f'{service_type}_supplyed_demands_without_delta']
        if self.buildings_old_values is not None:
            for col in buildsing_delta_cols:
                d = self.buildings_old_values[col.split('_delta')[0]].sub(self.new_Provisions[service_type]['buildings'][col.split('_delta')[0]], fill_value = 0)
                d = d.loc[self.new_Provisions[service_type]['buildings'].index]
                self.new_Provisions[service_type]['buildings'][col] =  d
        if self.services_old_values is not None:
            for col in services_delta_cols:
                d =  self.services_old_values[col.split('_delta')[0]].sub(self.new_Provisions[service_type]['services'][col.split('_delta')[0]], fill_value = 0) 
                d = d.loc[self.new_Provisions[service_type]['services'].index]
                self.new_Provisions[service_type]['services'][col] = d
        return self.new_Provisions[service_type]['buildings'], self.new_Provisions[service_type]['services'] 
    
    @staticmethod
    def _get_nk_distances(nk_dists, loc):
        target_nodes = loc.index
        source_node = loc.name
        distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]

        return pd.Series(data = distances, index = target_nodes)
    
    def _declare_varables(self, loc):
        name = loc.name
        nans = loc.isna()
        index = nans[~nans].index
        t = pd.Series([pulp.LpVariable(name = f"route_{name}_{I}", lowBound=0, cat = "Integer") for I in index], index)
        loc[~nans] = t
        return loc

    @staticmethod
    def _provision_matrix_transform(destination_matrix, 
                                    services, 
                                    buildings):
        def subfunc(loc):
            try:
                return [{"house_id":int(k),"demand":int(v), "service_id": int(loc.name)} for k,v in loc.to_dict().items()]
            except:
                return np.NaN
        def subfunc_geom(loc):
            return shapely.geometry.LineString((buildings['geometry'][loc['house_id']],services['geometry'][loc['service_id']]))

        buildings.geometry = buildings.centroid
        services.geometry = services.centroid
        flat_matrix = destination_matrix.transpose().apply(lambda x: subfunc(x[x>0]), result_type = "reduce")
        distribution_links = gpd.GeoDataFrame(data = [item for sublist in list(flat_matrix) for item in sublist])
        sel = distribution_links['house_id'].isin(buildings.index.values) & distribution_links['service_id'].isin(services.index.values)
        sel = distribution_links.loc[sel[sel].index.values]
        distribution_links['geometry'] = sel.apply(lambda x: subfunc_geom(x), axis = 1)
        return distribution_links

    def _provision_loop_linear(self, houses_table, services_table, distance_matrix, selection_range, destination_matrix, service_type): 
        select = distance_matrix[distance_matrix.iloc[:] <= selection_range]
        select = select.apply(lambda x: 1/(x+1), axis = 1)

        select = select.loc[:, ~select.columns.duplicated()].copy(deep = True)
        select = select.loc[~select.index.duplicated(),: ].copy(deep = True) 

        variables = select.apply(lambda x: self._declare_varables(x), axis = 1)

        prob = pulp.LpProblem("problem", pulp.LpMaximize)
        for col in variables.columns:
            t = variables[col].dropna().values
            if len(t) > 0: 
                prob +=(pulp.lpSum(t) <= houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'][col],
                        f"sum_of_capacities_{col}")
            else: pass

        for index in variables.index:
            t = variables.loc[index].dropna().values
            if len(t) > 0:
                prob +=(pulp.lpSum(t) <= services_table['capacity_left'][index],
                        f"sum_of_demands_{index}")
            else:pass
        costs = []
        for index in variables.index:
            t = variables.loc[index].dropna()
            t = t * select.loc[index].dropna()
            costs.extend(t)
        prob +=(pulp.lpSum(costs),
                "Sum_of_Transporting_Costs" )
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        to_df = {}
        for var in prob.variables():
            t = var.name.split('_')
            try:
                to_df[int(t[1])].update({int(t[2]): var.value()})
            except ValueError: 
                print(t)
                pass
            except:
                to_df[int(t[1])] = {int(t[2]): var.value()}
                
        result = pd.DataFrame(to_df).transpose()
        result = result.join(pd.DataFrame(0,
                                          columns = list(set(set(destination_matrix.columns) - set(result.columns))),
                                          index = destination_matrix.index), how = 'outer')
        result = result.fillna(0)
        destination_matrix = destination_matrix + result
        axis_1 = destination_matrix.sum(axis = 1)
        axis_0 = destination_matrix.sum(axis = 0)
        services_table['capacity_left'] = services_table['capacity'].subtract(axis_1,fill_value = 0)
        houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'] = houses_table[f'{service_type}_service_demand_value_{self.valuation_type}'].subtract(axis_0,fill_value = 0)

        distance_matrix = distance_matrix.drop(index = services_table[services_table['capacity_left'] == 0].index.values,
                                        columns = houses_table[houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'] == 0].index.values,
                                        errors = 'ignore')
        
        selection_range += selection_range
        if len(distance_matrix.columns) > 0 and len(distance_matrix.index) > 0:
            return self._provision_loop_linear(houses_table, services_table, distance_matrix, selection_range, destination_matrix, service_type)
        else: 
            print(houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'].sum(), services_table['capacity_left'].sum(),selection_range)
            return destination_matrix

    def _provision_loop_gravity(self, houses_table, services_table, distance_matrix, selection_range, destination_matrix, service_type, temp_destination_matrix = None):
        def _calculate_flows_y(loc):
            c = services_table.loc[loc.name]['capacity_left']
            d = houses_table.loc[loc.index][f'{service_type}_service_demand_left_value_{self.valuation_type}']
            p = d/loc
            p = p/p.sum()
            if p.sum() == 0:
                return loc
            else:
                rng = np.random.default_rng(seed = 0)
                r = pd.Series(0, p.index)
                choice = np.unique(rng.choice(p.index , int(c), p = p.values), return_counts = True)
                choice = r.add(pd.Series(choice[1], choice[0]), fill_value = 0)
            
                return choice

        def _balance_flows_to_demands(loc):
            d = houses_table.loc[loc.name][f'{service_type}_service_demand_left_value_{self.valuation_type}']
            loc = loc[loc > 0]
            if loc.sum() > 0:
                p = loc / loc.sum()
                rng = np.random.default_rng(seed = 0)
                r = pd.Series(0, p.index)
                choice = np.unique(rng.choice(p.index , int(d), p = p.values), return_counts = True)
                choice = r.add(pd.Series(choice[1], choice[0]), fill_value = 0)
                choice = pd.Series(data = np.minimum(loc.sort_index().values,
                                                    choice.sort_index().values), 
                                index = loc.sort_index().index)
                return choice
            else:
                return loc
                
        temp_destination_matrix = distance_matrix.apply(lambda x: _calculate_flows_y(x[x <= selection_range]), axis = 1)
        temp_destination_matrix = temp_destination_matrix.fillna(0)
        temp_destination_matrix = temp_destination_matrix.apply(lambda x: _balance_flows_to_demands(x))
        temp_destination_matrix = temp_destination_matrix.fillna(0)
        destination_matrix = destination_matrix.add(temp_destination_matrix, fill_value = 0)
        axis_1 = destination_matrix.sum(axis = 1)
        axis_0 = destination_matrix.sum(axis = 0)
        services_table['capacity_left'] = services_table['capacity'].subtract(axis_1,fill_value = 0)
        houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'] = houses_table[f'{service_type}_service_demand_value_{self.valuation_type}'].subtract(axis_0,fill_value = 0)

        distance_matrix = distance_matrix.drop(index = services_table[services_table['capacity_left'] == 0].index.values,
                                            columns = houses_table[houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'] == 0].index.values,
                                            errors = 'ignore')
        selection_range += selection_range
        if len(distance_matrix.columns) > 0 and len(distance_matrix.index) > 0:
            
            return self._provision_loop_gravity(houses_table, services_table, distance_matrix, selection_range, destination_matrix, service_type,temp_destination_matrix)                              
        else: 
            print(houses_table[f'{service_type}_service_demand_left_value_{self.valuation_type}'].sum(), services_table['capacity_left'].sum(),selection_range)
            return destination_matrix

