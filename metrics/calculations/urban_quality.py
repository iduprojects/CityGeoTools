import geopandas as gpd
import pandas as pd
import json
import numpy as np
import networkx as nx

from .utils import get_links
from .base_method import BaseMethod

class Urban_Quality(BaseMethod):

    def __init__(self, city_model):
        '''
        returns urban quality index and raw data for it
        metric calculates different quantity parameters of urban environment (share of emergent houses, number of cultural objects, etc.)
        and returns rank of urban quality for each city block (from 1 to 10, and 0 is for missing data)
        >>> Urban_Quality(city_model).get_urban_quality()

        '''
        BaseMethod.__init__(self, city_model)
        self.buildings = city_model.Buildings.copy()
        self.services = city_model.Services.copy()
        self.blocks = city_model.Blocks.copy()
        self.greenery = city_model.RecreationalAreas.copy()
        self.city_crs = city_model.city_crs
        
        self.main_services_id = [172, 130, 131, 132, 163, 183, 174, 165, 170, 176, 175, 161, 173, 88, 124, 51, 47, 46, 45, 41, 40,
        39, 37, 35, 55, 34, 29, 27, 26, 20, 18, 17, 14, 13, 11, 33, 62, 65, 66, 121, 120, 113, 106, 102, 97, 94, 93, 92, 90, 189,
        86, 85, 84, 83, 82, 79, 78, 77, 69, 67, 125, 190]
        self.street_services_id = [31, 181, 88, 20, 25, 87, 28, 30, 60, 27, 86, 18, 90, 62, 83, 47, 17, 63, 39,
        22, 163, 84, 32, 15, 24, 26, 46, 11, 53, 190, 172, 89, 92, 29, 48, 81, 161, 162, 147, 165, 148, 170, 168, 37, 178,
        54, 179, 51, 156, 169, 176]
        self.drive_graph = nx.Graph(((u, v, e) for u,v,e in city_model.MobilityGraph.edges(data=True) if e['type'] == 'car'))
        
    def _ind1(self):

        local_blocks = self.blocks.copy()
        local_buildings = self.buildings.copy()
        
        normal_buildings = local_buildings.dropna(subset=['is_emergency'])
        emergent_buildings = normal_buildings.query('is_emergency')

        local_blocks['emergent_living_area'] = local_blocks.merge(emergent_buildings.groupby(['block_id'])\
            .sum().reset_index(), how='left', left_on='id', right_on='block_id')['living_area']
        local_blocks['living_area'] = local_blocks.merge(normal_buildings.groupby(['block_id'])\
            .sum().reset_index(), how='left', left_on='id', right_on='block_id')['living_area']
        local_blocks.loc[(local_blocks.living_area.notnull() & local_blocks.emergent_living_area.isnull()), 'emergent_living_area'] = 0
        local_blocks['IND_data'] = (local_blocks['living_area'] - local_blocks['emergent_living_area']) / local_blocks['living_area']

        local_blocks['IND'] = pd.cut(local_blocks['IND_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND'] = pd.to_numeric(local_blocks['IND']).fillna(0).astype(int)
        print('Indicator 1 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind2(self):

        local_blocks = self.blocks.copy()
        local_buildings = self.buildings.copy()

        normal_buildings = local_buildings.dropna(subset=['central_heating', 'central_hotwater'])
        normal_buildings = normal_buildings.dropna(subset=['central_electro', 'central_gas'], how='all')

        accomodated_buildings = normal_buildings.query('central_heating & central_hotwater & (central_electro | central_gas)')

        normal_buildings_in_blocks = normal_buildings.groupby('block_id').sum().reset_index()
        accomodated_buildings_in_blocks = accomodated_buildings.groupby('block_id').sum().reset_index()
        local_blocks['normal_living_area'] = local_blocks.merge(normal_buildings_in_blocks,\
            how='left', left_on='id', right_on='block_id')['living_area']
        local_blocks['accomodated_living_area'] = local_blocks.merge(accomodated_buildings_in_blocks,\
            how='left', left_on='id', right_on='block_id')['living_area']

        local_blocks.loc[(local_blocks.normal_living_area.notnull() & local_blocks.accomodated_living_area.isnull()), 'accomodated_living_area'] = 0
        local_blocks['IND_data'] = local_blocks['accomodated_living_area'] / local_blocks['normal_living_area']

        local_blocks['IND'] = pd.cut(local_blocks['IND_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND'] = pd.to_numeric(local_blocks['IND']).fillna(0).astype(int)
        print('Indicator 2 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind4(self):

        houses = self.buildings.copy()
        houses = houses[houses['is_living'] == True]
        local_blocks = self.blocks.copy()
        modern_houses = houses.query('1956 <= building_year')

        modern_houses_diversity = modern_houses.groupby('block_id').agg({'id':'size', 'project_type':'nunique'}).reset_index()
        modern_houses_diversity['total_count'] = modern_houses_diversity.merge\
            (houses.groupby('block_id').count().reset_index())['id']
        modern_houses_diversity['project_type'].replace(0, 1, inplace=True)
        modern_houses_diversity['weighted_diversity'] = (modern_houses_diversity.project_type / modern_houses_diversity.total_count)

        local_blocks['IND_data'] = local_blocks.merge(modern_houses_diversity,\
            how='left', left_on='id', right_on='block_id')['weighted_diversity']
        local_blocks['IND'] = pd.cut(local_blocks['IND_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND'] = pd.to_numeric(local_blocks['IND']).fillna(0).astype(int)
        print('Indicator 4 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind5(self):

        local_blocks = self.blocks.copy()
        houses = self.buildings.copy()
        houses = houses[houses['is_living'] == True]
        local_services = self.services.copy()
        local_services = local_services[local_services['city_service_type_id'].isin(self.main_services_id)]

        count_in_blocks = local_services.groupby('block_id').count().reset_index()
        count_in_blocks['IND_data'] = count_in_blocks['id']/count_in_blocks['building_id']
        count_in_blocks = count_in_blocks[(count_in_blocks.IND_data != np.inf) &\
            (count_in_blocks.block_id.isin(pd.unique(houses.block_id)))]

        local_blocks['IND_data'] = local_blocks.merge(count_in_blocks,\
            how='left', left_on='id', right_on='block_id')['IND_data']

        local_blocks['IND'] = pd.cut(local_blocks['IND_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND'] = pd.to_numeric(local_blocks['IND']).fillna(0).astype(int)
        print('Indicator 5 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind10(self):

        local_blocks = self.blocks.copy()
        local_buildings = self.buildings.copy()
        local_services = self.services.copy()
        local_services = local_services[local_services['city_service_type_id'].isin(self.street_services_id)]
        walk_links = get_links(self.drive_graph, self.city_crs)

        walk_links['geometry'] = walk_links.geometry.buffer(40)
        walk_links['link_id'] = walk_links.index

        #Arguments
        links_with_objects = gpd.sjoin(walk_links, local_buildings[['geometry', 'id']], how='inner')
        walk_links['n_buildings'] = walk_links.merge(links_with_objects.groupby('link_id').count().reset_index(),\
         on='link_id', how='left')['id'].dropna()

        links_with_objects = gpd.sjoin(walk_links, local_services[['geometry', 'city_service_type', 'id']], how='inner')
        walk_links['n_services'] = walk_links.merge(links_with_objects.groupby('link_id').count().reset_index(),\
         on='link_id', how='left')['id'].fillna(0)
        walk_links['n_types_services'] = walk_links.merge(links_with_objects.groupby('link_id').nunique()['city_service_type']\
        .reset_index(), how='left')['city_service_type'].fillna(0)

        #Indicators
        N_types = len(pd.unique(local_services.city_service_type))
        walk_links['variety'] = (walk_links['n_types_services']/N_types)
        walk_links['density'] = ((walk_links['n_services']/walk_links['length_meter'])*100)
        walk_links['saturation'] = (walk_links['n_services']/walk_links['n_buildings'])

        # Maturity
        walk_links['maturity'] = (0.5*walk_links['variety'] + 0.25*walk_links['density'] + 0.25*walk_links['saturation'])

        walk_links_with_blocks = gpd.sjoin(local_blocks[['geometry', 'id']], walk_links[['geometry', 'maturity']], how='inner')
        local_blocks['IND_data'] = local_blocks.merge(walk_links_with_blocks.groupby('id').mean().reset_index(), how='left')['maturity']

        local_blocks['IND'] = pd.cut(local_blocks['IND_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND'] = pd.to_numeric(local_blocks['IND']).fillna(0).astype(int)
        print('Indicator 10 done')
        return local_blocks['IND'], local_blocks['IND_data']
    def _ind14(self):
        local_blocks = self.blocks.copy()
        local_greenery = self.greenery.copy()
        local_greenery = local_greenery.explode(ignore_index=True)
        local_greenery = local_greenery[local_greenery.geometry.type =="Polygon"]

        local_blocks['area'] = local_blocks.area
        greenery_in_blocks = gpd.overlay(local_blocks, local_greenery, how='intersection')
        greenery_in_blocks['green_area'] = greenery_in_blocks.area
        share_of_green = greenery_in_blocks.groupby('block_id').sum('green_area').reset_index()
        share_of_green['share'] = share_of_green['green_area'] / share_of_green['area']

        local_blocks['IND_14_data']  = local_blocks.merge(share_of_green, how='left')['share'] 
        local_blocks['IND_14'] = pd.cut(local_blocks['IND_14_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND_14'] = pd.to_numeric(local_blocks['IND_14']).fillna(0).astype(int)
        print('Indicator 14 done')
        return local_blocks['IND_14'], local_blocks['IND_14_data']

    def _ind15(self):

        local_blocks = self.blocks.copy()
        local_greenery = self.greenery.copy()

        if len(local_greenery.vegetation_index) == local_greenery.vegetation_index.isna().sum():
            local_blocks['IND_15_data'] = 0
            local_blocks['IND_15'] = 0
            print('IND15: vegetation index is not loaded')
            return local_blocks['IND_15'], local_blocks['IND_15_data']

        local_greenery = local_greenery.explode(ignore_index=True)
        local_greenery = local_greenery[local_greenery.geometry.type =="Polygon"]

        local_blocks['area'] = local_blocks.area
        greenery_in_blocks = gpd.overlay(local_blocks, local_greenery, how='intersection')
        greenery_in_blocks['green_area'] = greenery_in_blocks.area
        share_of_green = greenery_in_blocks.groupby(['block_id', 'vegetation_index']).sum('green_area').reset_index()
        share_of_green['share'] = share_of_green['green_area'] / share_of_green['area']
        share_of_green['vw'] = share_of_green.vegetation_index.astype(float) * share_of_green.share.astype(float)

        share_of_green_grouped = share_of_green.groupby('block_id').sum().reset_index()
        share_of_green_grouped['quality'] = share_of_green_grouped.vw / share_of_green_grouped.share

        local_blocks['IND_15_data']  = local_blocks.merge(share_of_green_grouped, how='left')['quality'] 

        local_blocks['IND_15'] = pd.cut(local_blocks['IND_15_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND_15'] = pd.to_numeric(local_blocks['IND_15']).fillna(0).astype(int)
        print('Indicator 15 done')
        return local_blocks['IND_15'], local_blocks['IND_15_data']

    def _ind17(self):

        local_blocks = self.blocks.copy()
        local_services = self.services.copy()
        local_services = local_services[local_services['city_service_type_id'].isin(self.main_services_id)]
        local_greenery = self.greenery.copy()
        local_greenery = local_greenery.explode(ignore_index=True)
        local_greenery = local_greenery[local_greenery.geometry.type =="Polygon"]

        greenery_in_blocks = gpd.overlay(local_blocks, local_greenery[['geometry', 'service_code', 'block_id']], how='intersection')
        greenery_in_blocks['green_area'] = greenery_in_blocks.area

        services_in_greenery = gpd.sjoin(greenery_in_blocks, local_services['geometry'].reset_index(), how='inner')
        services_in_greenery = services_in_greenery.groupby(['id', 'block_id', 'green_area']).count().reset_index()
        services_in_greenery = services_in_greenery.groupby('block_id').sum().reset_index()
        services_in_greenery['weighted_service_count'] = services_in_greenery['id'] / services_in_greenery['green_area']

        local_blocks['IND_data'] = local_blocks.merge(services_in_greenery,\
                left_on='id', right_on='block_id', how='left')['weighted_service_count']

        local_blocks['IND'] = pd.cut(local_blocks['IND_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND'] = pd.to_numeric(local_blocks['IND']).fillna(0).astype(int)
        print('Indicator 17 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind22(self):

        local_blocks = self.blocks.copy()
        local_okn = self.services.copy()
        local_okn = local_okn[local_okn['service_code'] =='culture_object']

        if len(local_okn) == 0:
            local_blocks['IND_22_data'] = 0
            local_blocks['IND_22'] = 0
            print('IND22: culture objects are not loaded')
            return local_blocks['IND_22'], local_blocks['IND_22_data']

        local_blocks['area'] = local_blocks.area
        okn_in_blocks = gpd.sjoin(local_okn[['geometry', 'service_code']], local_blocks, how='inner')
        local_blocks['n_okn'] = local_blocks.merge(okn_in_blocks.groupby('id').count().reset_index(), on='id', how='left')['service_code']
        local_blocks['IND_data'] = (local_blocks['n_okn'] / local_blocks['area'])

        local_blocks['IND'] = pd.cut(local_blocks['IND_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND'] = pd.to_numeric(local_blocks['IND']).fillna(0).astype(int)
        print('Indicator 22 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind23(self):

        local_blocks = self.blocks.copy()
        local_services = self.services.copy()
        local_buildings = self.buildings.copy()
        #add total building area
        local_buildings['storeys_count'].fillna(1.0, inplace=True)
        local_buildings['building_area'] = local_buildings['basement_area'] * local_buildings['storeys_count']
        #calculate MXI
        sum_grouper = local_buildings.groupby(["block_id"]).sum().reset_index()
        sum_grouper['MXI'] = sum_grouper["living_area"] / sum_grouper["building_area"]
        sum_grouper = sum_grouper.query('0 < MXI <= 0.8')
        #filter commercial services
        local_services = local_services[local_services['city_service_type_id'].isin(self.main_services_id)]
        #calculate free area for commercial services
        sum_grouper['commercial_area'] = sum_grouper['building_area'] - sum_grouper['living_area'] - sum_grouper['building_area']*0.1
        #calculate number of commercial services per block
        local_blocks['n_services'] = local_blocks.merge(local_services.groupby('block_id').count().reset_index(),\
            left_on='id', right_on='block_id', how='left')['service_code']
        local_blocks['commercial_area'] = local_blocks.merge(sum_grouper, left_on='id', right_on='block_id',\
             how='left')['commercial_area']
        #calculate commercial diversity
        local_blocks['IND_data'] = (local_blocks['n_services'] / local_blocks['commercial_area'])

        local_blocks['IND'] = pd.cut(local_blocks['IND_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND'] = pd.to_numeric(local_blocks['IND']).fillna(0).astype(int)
        print('Indicator 23 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind30(self):

        local_blocks = self.blocks.copy()
        Provision_class = City_Provisions(self.city_model, service_types = ["kindergartens"], valuation_type = "normative", year = 2022,\
            user_provisions=None, user_changes_buildings=None, user_changes_services=None, user_selection_zone=None, service_impotancy=None)
        kindergartens_provision = Provision_class.get_provisions()
        kindergartens_provision = gpd.GeoDataFrame.from_features(kindergartens_provision['houses']['features'])

        local_blocks['IND_data'] = local_blocks.merge(kindergartens_provision.groupby('block_id').mean().reset_index(),\
            left_on='id', right_on='block_id', how='left')['kindergartens_provison_value']

        local_blocks['IND'] = pd.cut(local_blocks['IND_data'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
        local_blocks['IND'] = pd.to_numeric(local_blocks['IND']).fillna(0).astype(int)
        print('Indicator 30 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def get_urban_quality(self):

        urban_quality = self.blocks.copy().to_crs(4326)
        urban_quality_raw = self.blocks.copy().to_crs(4326)

        urban_quality['ind1'], urban_quality_raw['ind1'] = self._ind1()
        urban_quality['ind2'], urban_quality_raw['ind2'] = self._ind2()
        urban_quality['ind4'], urban_quality_raw['ind4'] = self._ind4()
        urban_quality['ind5'], urban_quality_raw['ind5'] = self._ind5()
        urban_quality['ind10'], urban_quality_raw['idn10'] = self._ind10()
        #urban_quality['ind11'], urban_quality_raw['ind11'] = self._ind_11() #too long >15 min
        #urban_quality['ind13'], urban_quality_raw['ind13'] = self._ind_13() #recreational areas problem
        urban_quality['ind14'], urban_quality_raw['ind14'] = self._ind14()
        urban_quality['ind15'], urban_quality_raw['ind15'] = self._ind15()
        urban_quality['ind17'], urban_quality_raw['ind17'] = self._ind17()
        #urban_quality['ind18'], urban_quality_raw['ind18'] = self._ind18() #recreational areas problem
        #urban_quality['ind20'], urban_quality_raw['ind20'] = self._ind20() #too much in RAM
        urban_quality['ind22'], urban_quality_raw['ind22'] = self._ind22()
        urban_quality['ind23'], urban_quality_raw['ind23'] = self._ind23()
        #urban_quality['ind25'], urban_quality_raw['ind25'] = self._ind25() #no crosswalks provision in database
        #urban_quality['ind30'], urban_quality_raw['ind30'] = self._ind30(city_model) #kindergartens not loaded
        #urban_quality['ind32'], urban_quality_raw['ind32'] = self._ind32() #no stops provision in database

        urban_quality = urban_quality.replace(0, np.NaN)
        urban_quality['urban_quality_value'] = urban_quality.filter(regex='ind.*').mean(axis=1).round(0)
        urban_quality = urban_quality.fillna(0)
        
        return {'urban_quality': json.loads(urban_quality.to_json()),
                'urban_quality_data': json.loads(urban_quality_raw.to_json())}