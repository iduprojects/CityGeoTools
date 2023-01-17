import geopandas as gpd
import pandas as pd
import json
import numpy as np
import networkx as nx
import os
import io
import requests

from .utils import get_links
from .base_method import BaseMethod
from .provision import CityProvision

class UrbanQuality(BaseMethod):

    def __init__(self, city_model):
        '''
        returns urban quality index and raw data for it
        metric calculates different quantity parameters of urban environment (share of emergent houses, number of cultural objects, etc.)
        and returns rank of urban quality for each city block (from 1 to 10, and 0 is for missing data)
        metric is based on urban quality index of Strelka KB (https://xn----dtbcccdtsypabxk.xn--p1ai/#/methodology)
        >>> UrbanQuality(city_model).get_urban_quality()

        '''
        BaseMethod.__init__(self, city_model)
        self.buildings = city_model.Buildings.copy()
        self.services = city_model.Services.copy()
        self.blocks = city_model.Blocks.copy()
        self.greenery = city_model.RecreationalAreas.copy()
        self.city_crs = city_model.city_crs
        self.city_name = city_model.city_name
        self.engine = city_model.engine
        self.file_server = os.environ['PROVISIONS_DATA_FILE_SERVER']
        
        self.main_services = ['cemeteries', 'post_boxes', 'recreational_areas', 'garbage_containers', 'child_goods', 'diagnostic_center',
        'book_store', 'copy_center', 'toy_store', 'art_goods', 'amusement_park', 'beach', 'sanatorium', 'holiday_goods', 'stops',
        'pharmacies', 'optics', 'churches', 'beauty_salons', 'outposts', 'conveniences', 'markets', 'supermarkets', 'hardware_stores',
        'instrument_stores', 'electronic_stores', 'clothing_stores', 'sporting_stores', 'flower_stores', 'pet_shops', 'veterinaries',
        'museums', 'shopping_centers', 'cinemas', 'swimming_pools', 'saunas', 'bakeries', 'garbage_dumps', 'nursery',
        'ambulance_stations', 'stadiums', 'rituals', 'traffic_service', 'psychology', 'morgue', 'fire_stations', 'aquaparks',
        'recycling_stations', 'dog_playgrounds', 'housing_services', 'animal_shelter', 'pet_market']
        self.street_services = ['gas_stations', 'car_showroom', 'child_goods', 'child_teenager_club', 'book_store', 'toy_store', 'art_goods',
        'amusement_park', 'car_rental', 'wedding_agency', 'sport_section', 'holiday_goods', 'pharmacies', 'optics', 'repair_stations',
        'beauty_salons', 'conveniences', 'markets', 'supermarkets', 'hardware_stores', 'instrument_stores', 'clothing_stores', 
        'tobacco_stores', 'sporting_stores', 'jewelry_stores', 'flower_stores', 'pet_shops', 'libraries', 'theaters', 'museums',
        'shopping_centers', 'cinemas', 'swimming_pools', 'saunas', 'sport_centers', 'bars', 'bakeries', 'cafes', 'restaurants',
        'fastfoods', 'visa_centers', 'bookmaker_offices', 'limousine_rental', 'spas', 'art_spaces', 'aquaparks',
        'scooter_rental', 'art_gallery', 'circus', 'sport_clubs', 'pet_market']
        self.drive_graph = nx.Graph(((u, v, e) for u,v,e in city_model.MobilityGraph.edges(data=True) if e['type'] == 'car'))

    @staticmethod
    def _ind_ranking(data_series):
        return pd.to_numeric(pd.cut(data_series, 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False))

    @staticmethod
    def _greenery_exploder(greenery):
        greenery = greenery.explode(ignore_index=True)
        return greenery[greenery.geometry.type =="Polygon"]
        
    def _ind1(self):
        '''
        calculates share of living area of emergency houses in total living area by blocks
        '''
        local_blocks = self.blocks.copy()
        local_buildings = self.buildings.copy()
        
        normal_buildings = local_buildings.dropna(subset=['is_emergency'])
        emergent_buildings = normal_buildings.query('is_emergency')

        local_blocks = local_blocks.join(normal_buildings[['block_id', 'living_area']].groupby('block_id').sum('living_area'), on='id')
        local_blocks = local_blocks.join(emergent_buildings[['block_id', 'living_area']].groupby('block_id').sum('living_area'), on='id', rsuffix='_emergent')

        local_blocks.loc[(local_blocks.living_area.notnull() & local_blocks.living_area_emergent.isnull()), 'living_area_emergent'] = 0
        local_blocks['IND_data'] = (local_blocks['living_area'] - local_blocks['living_area_emergent']) / local_blocks['living_area']
        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 1 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind2(self):
        '''
        calculates share of living area of houses with communal insfrastructure in total living area by blocks
        '''
        local_blocks = self.blocks.copy()
        local_buildings = self.buildings.copy()

        normal_buildings = local_buildings.dropna(subset=['central_heating', 'central_hotwater'])
        normal_buildings = normal_buildings.dropna(subset=['central_electro', 'central_gas'], how='all')

        accomodated_buildings = normal_buildings.query('central_heating & central_hotwater & (central_electro | central_gas)')

        local_blocks = local_blocks.join(normal_buildings[['block_id', 'living_area']].groupby('block_id').sum('living_area'), on='id')
        local_blocks = local_blocks.join(accomodated_buildings[['block_id', 'living_area']].groupby('block_id').sum('living_area'), on='id', rsuffix='_accomodated')
       

        local_blocks.loc[(local_blocks.living_area.notnull() & local_blocks.living_area_accomodated.isnull()), 'living_area_accomodated'] = 0
        local_blocks['IND_data'] = local_blocks['living_area_accomodated'] / local_blocks['living_area']

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 2 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind4(self):
        '''
        calculates diversity of project types of houses by blocks
        '''
        local_blocks = self.blocks.copy()
        houses = self.buildings.copy()
        houses = houses[houses['is_living'] == True]
        common_projects = list(houses.groupby('project_type').count()['id'].drop('Индивидуальный').sort_values(ascending=False)[:2].index)
        local_blocks = local_blocks.join(houses[houses.project_type.isin(common_projects)][['block_id', 'id']]\
            .groupby('block_id').count(), on='id', rsuffix='_common_count')
        local_blocks = local_blocks.join(houses[['block_id', 'id']]\
            .groupby('block_id').count(), on='id', rsuffix='_total_count')
        local_blocks['IND_data'] = local_blocks['id_common_count'] / local_blocks['id_total_count']

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 4 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind5(self):
        '''
        calculates diversity of services in houses by blocks
        '''
        local_blocks = self.blocks.copy()
        houses = self.buildings.copy()
        houses = houses[houses['is_living'] == True]
        local_services = self.services.copy()
        local_services = local_services[local_services['service_code'].isin(self.main_services)]

        count_in_blocks = local_services[['block_id', 'building_id', 'id']].groupby('block_id').count().query('building_id > 0')
        count_in_blocks['IND_data'] = count_in_blocks['id']/count_in_blocks['building_id']
        local_blocks = local_blocks.join(count_in_blocks, on='id', rsuffix='_service')

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 5 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind10(self):
        '''
        calculates level of service development of streets by blocks
        '''
        local_blocks = self.blocks.copy()
        local_services = self.services.copy()
        local_services = local_services[local_services['service_code'].isin(self.street_services)]
        drive_links = get_links(self.drive_graph, self.city_crs)

        drive_links['geometry'] = drive_links.geometry.buffer(40)
        drive_links['link_id'] = drive_links.index

        #Arguments
        links_with_data = gpd.sjoin(drive_links, local_services, how='inner')
        drive_links = drive_links.join(links_with_data[['link_id', 'id', 'building_id', 'service_code']].groupby('link_id').nunique(), on='link_id')
        drive_links = drive_links.dropna(subset='building_id')
        drive_links = drive_links.query('building_id > 0')
        drive_links = drive_links.fillna(0)
        #Indicators
        N_types = len(pd.unique(local_services.service_code))
        drive_links['variety'] = (drive_links['service_code']/N_types)
        drive_links['density'] = (drive_links['id']/(drive_links['length_meter']*100))
        drive_links['saturation'] = (drive_links['id']/drive_links['building_id'])
        # Maturity
        drive_links['IND_data'] = (0.5*drive_links['variety'] + 0.25*drive_links['density'] + 0.25*drive_links['saturation'])

        drive_links = gpd.sjoin(local_blocks, drive_links[['geometry', 'IND_data']], how='inner')
        drive_links[['id', 'IND_data']].groupby('id').mean()
        local_blocks = local_blocks.join(drive_links[['id', 'IND_data']].groupby('id').mean(), on='id')

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 10 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind13(self):
        '''
        calculates recreational areas' usage activity by blocks
        '''
        local_blocks = self.blocks
        Provisions_class = CityProvision(self.city_model,
                            service_types = ['recreational_areas'],
                            valuation_type = "normative",
                            year = 2022, 
                            user_changes_buildings = None,
                            user_changes_services = None,
                            user_provisions = None,
                            user_selection_zone = None,
                            service_impotancy = None)
        local_provision = Provisions_class.get_provisions().services
        local_provision['IND_data'] = local_provision['service_load'] / local_provision['capacity']
        local_blocks = local_blocks.join(local_provision[['block_id', 'IND_data']].groupby('block_id').mean(), on='id')
        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])

        print('Indicator 13 done')
        return local_blocks['IND'], local_blocks['IND_data']    

    def _ind14(self):
        '''
        calculates share of greenery in total area of blocks
        '''
        local_blocks = self.blocks.copy()
        local_greenery = self.greenery.copy()
        local_greenery = self._greenery_exploder(local_greenery)

        local_blocks['area'] = local_blocks.area
        greenery_in_blocks = gpd.overlay(local_blocks, local_greenery, how='intersection')
        greenery_in_blocks['green_area'] = greenery_in_blocks.area
        share_of_green = greenery_in_blocks[['block_id', 'area', 'green_area']].groupby('block_id').sum()
        share_of_green['IND_data'] = share_of_green['green_area'] / share_of_green['area']
        local_blocks = local_blocks.join(share_of_green['IND_data'], on='id')

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 14 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind15(self):
        '''
        calculates quality of greenery of blocks
        '''
        local_blocks = self.blocks.copy()
        local_greenery = self.greenery.copy()

        if local_greenery.vegetation_index.isna().all():
            local_blocks['IND_data'] = None
            local_blocks['IND'] = None
            print('IND15: vegetation index is not loaded')
            return local_blocks['IND'], local_blocks['IND_data']

        local_greenery = self._greenery_exploder(local_greenery)
        local_blocks['area'] = local_blocks.area
        greenery_in_blocks = gpd.overlay(local_blocks, local_greenery, how='intersection')
        greenery_in_blocks['green_area'] = greenery_in_blocks.area
        share_of_green = greenery_in_blocks.groupby(['block_id', 'vegetation_index']).sum('green_area').reset_index()
        share_of_green['share'] = share_of_green['green_area'] / share_of_green['area']

        share_of_green['vw'] = share_of_green.vegetation_index * share_of_green.share

        share_of_green_grouped = share_of_green[['block_id', 'vw', 'share']].groupby('block_id').sum()
        share_of_green_grouped['IND_data'] = share_of_green_grouped.vw / share_of_green_grouped.share

        local_blocks  = local_blocks.join(share_of_green_grouped['IND_data'], on='id')

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 15 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind17(self):
        '''
        calculates diversity of services in recreational areas by blocks
        '''
        local_blocks = self.blocks.copy()
        local_services = self.services.copy()
        local_services = local_services[local_services['service_code'].isin(self.main_services)]
        local_greenery = self.greenery.copy()
        local_greenery = local_greenery.explode(ignore_index=True)
        local_greenery = local_greenery[local_greenery.geometry.type =="Polygon"]

        greenery_in_blocks = gpd.overlay(local_blocks, local_greenery[['geometry', 'service_code', 'block_id']], how='intersection')
        greenery_in_blocks['green_area'] = greenery_in_blocks.area

        services_in_greenery = gpd.sjoin(greenery_in_blocks, local_services['geometry'].reset_index(), how='inner')
        services_in_greenery = services_in_greenery.groupby(['id', 'block_id', 'green_area']).count().reset_index()
        services_in_greenery = services_in_greenery.groupby('block_id').sum()
        services_in_greenery['IND_data'] = services_in_greenery['id'] / services_in_greenery['green_area']

        local_blocks = local_blocks.join(services_in_greenery['IND_data'], on='id')

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 17 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind18(self):
        '''
        calculates provision of recreational areas by blocks
        '''
        local_blocks = self.blocks
        Provisions_class = CityProvision(self.city_model,
                                    service_types = ['recreational_areas'],
                                    valuation_type = "normative",
                                    year = 2022, 
                                    user_changes_buildings = None,
                                    user_changes_services = None,
                                    user_provisions = None,
                                    user_selection_zone = None,
                                    service_impotancy = None)
        local_provision = Provisions_class.get_provisions().buildings
        local_blocks = local_blocks.join(local_provision[['block_id', 'recreational_areas_provison_value']].groupby('block_id').mean(), on='id')
        local_blocks.rename(columns={'recreational_areas_provison_value':'IND_data'}, inplace=True)

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 18 done')
        return local_blocks['IND'], local_blocks['IND_data']    

    def _ind22(self):
        '''
        calculates amount of culture objects in blocks
        '''
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
        local_blocks = local_blocks.join(okn_in_blocks[['id', 'service_code']].groupby('id').count(), on='id')
        local_blocks['IND_data'] = local_blocks['service_code'] / local_blocks['area']

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 22 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind23(self):
        '''
        calculates level of service development of commercial areas in houses by blocks
        '''
        local_blocks = self.blocks.copy()
        local_services = self.services.copy()
        local_buildings = self.buildings.copy()
        #add total building area
        local_buildings.dropna(subset='storeys_count', inplace=True)
        local_buildings['building_area'] = local_buildings['basement_area'] * local_buildings['storeys_count']
        #calculate MXI
        sum_grouper = local_buildings[['block_id', 'living_area', 'building_area']].groupby("block_id").sum()
        sum_grouper['MXI'] = sum_grouper["living_area"] / sum_grouper["building_area"]
        sum_grouper = sum_grouper.query('0 < MXI <= 0.8')
        #filter commercial services
        local_services = local_services[local_services['service_code'].isin(self.main_services)]
        #calculate free area for commercial services
        sum_grouper['commercial_area'] = sum_grouper['building_area'] - sum_grouper['living_area']
        #calculate number of commercial services per block
        local_blocks = local_blocks.join(local_services[['block_id', 'service_code']].groupby('block_id').count(), on='id')
        local_blocks = local_blocks.join(sum_grouper['commercial_area'], on='id')
        #calculate commercial diversity
        local_blocks['IND_data'] = local_blocks['service_code'] / local_blocks['commercial_area']

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 23 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind25(self):
        '''
        calculates provision of crosswalks as a measure of pedestrian safety
        '''
        local_blocks = self.blocks
        Provisions_class = CityProvision(self.city_model,
                                    service_types = ['crosswalks'],
                                    valuation_type = "normative",
                                    year = 2022, 
                                    user_changes_buildings = None,
                                    user_changes_services = None,
                                    user_provisions = None,
                                    user_selection_zone = None,
                                    service_impotancy = None)
        local_provision = Provisions_class.get_provisions().buildings
        local_blocks = local_blocks.join(local_provision[['block_id', 'crosswalks_provison_value']].groupby('block_id').mean(), on='id')
        local_blocks.rename(columns={'crosswalks_provison_value':'IND_data'}, inplace=True)

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 25 done')
        return local_blocks['IND'], local_blocks['IND_data']    

    def _ind30(self):
        '''
        calculates provision of kindergartens
        '''
        local_blocks = self.blocks
        Provisions_class = CityProvision(self.city_model,
                                    service_types = ['kindergartens'],
                                    valuation_type = "normative",
                                    year = 2022, 
                                    user_changes_buildings = None,
                                    user_changes_services = None,
                                    user_provisions = None,
                                    user_selection_zone = None,
                                    service_impotancy = None)
        local_provision = Provisions_class.get_provisions().buildings
        local_blocks = local_blocks.join(local_provision[['block_id', 'kindergartens_provison_value']].groupby('block_id').mean(), on='id')
        local_blocks.rename(columns={'kindergartens_provison_value':'IND_data'}, inplace=True)

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 30 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _ind32(self):
        '''
        calculates provision of public transport stops
        '''
        local_blocks = self.blocks
        Provisions_class = CityProvision(self.city_model,
                                    service_types = ['stops'],
                                    valuation_type = "normative",
                                    year = 2022, 
                                    user_changes_buildings = None,
                                    user_changes_services = None,
                                    user_provisions = None,
                                    user_selection_zone = None,
                                    service_impotancy = None)
        local_provision = Provisions_class.get_provisions().buildings
        local_blocks = local_blocks.join(local_provision[['block_id', 'stops_provison_value']].groupby('block_id').mean(), on='id')
        local_blocks.rename(columns={'stops_provison_value':'IND_data'}, inplace=True)

        local_blocks['IND'] = self._ind_ranking(local_blocks['IND_data'])
        print('Indicator 32 done')
        return local_blocks['IND'], local_blocks['IND_data']

    def _calculate_urban_quality(self):

        urban_quality = self.blocks.copy().to_crs(4326)

        urban_quality['ind1'], urban_quality['data_ind1'] = self._ind1()
        urban_quality['ind2'], urban_quality['data_ind2'] = self._ind2()
        urban_quality['ind4'], urban_quality['data_ind4'] = self._ind4()
        urban_quality['ind5'], urban_quality['data_ind5'] = self._ind5()
        urban_quality['ind10'], urban_quality['data_ind10'] = self._ind10()
        #urban_quality['ind11'], urban_quality['data_ind11'] = self._ind_11() #too long >15 min
        #urban_quality['ind13'], urban_quality['data_ind13'] = self._ind_13() #recreational areas problem
        urban_quality['ind14'], urban_quality['data_ind14'] = self._ind14()
        urban_quality['ind15'], urban_quality['data_ind15'] = self._ind15()
        urban_quality['ind17'], urban_quality['data_ind17'] = self._ind17()
        #urban_quality['ind18'], urban_quality['data_ind18'] = self._ind18() #recreational areas problem
        #urban_quality['ind20'], urban_quality['data_ind20'] = self._ind20() #too much in RAM
        urban_quality['ind22'], urban_quality['data_ind22'] = self._ind22()
        urban_quality['ind23'], urban_quality['data_ind23'] = self._ind23()
        #urban_quality['ind25'], urban_quality['data_ind25'] = self._ind25() #no crosswalks provision in database
        urban_quality['ind30'], urban_quality['data_ind30'] = self._ind30()
        #urban_quality['ind32'], urban_quality['data_ind32'] = self._ind32() #no stops provision in database

        urban_quality['urban_quality_value'] = urban_quality.filter(regex='^ind.*').mean(axis=1).round(0)
        
        return urban_quality

    def get_urban_quality(self):
        urban_quality = self._calculate_urban_quality()
        urban_quality = urban_quality.fillna(0)
        return json.loads(urban_quality.to_json())

    def get_urban_quality_context(self):
        urban_quality = self._calculate_urban_quality()
        indicators = pd.read_sql(f'''SELECT * FROM urban_quality.indicators_view''', con=self.engine)
        urban_quality = urban_quality.filter(regex='^ind.*').mean().round(0).reset_index()
        urban_quality.columns = ['indicator_id', 'indicator_value']
        urban_quality['indicator_id'] = urban_quality.indicator_id.str[3:].astype(int)
        urban_quality = indicators.merge(urban_quality, how='left')
        urban_quality['indicator_value'] = urban_quality.indicator_value.fillna(0).astype(int)
        values = []
        for space in pd.unique(urban_quality.space_name):
            values.append({space:json.loads(urban_quality[urban_quality['space_name'] == space][['indicator_id', 'indicator_value']].to_json(orient='records'))})
        description = []
        for index, row in indicators.iterrows():
            description.append({f'Indicator {row.indicator_id}': json.loads(row[['indicator_id', 'description']].to_json())})
        return {"urban_quality":{"values":values, "description":description}, 
            "rank": int(urban_quality.indicator_value.sum().astype(int)),
            "max_rank": int(len(urban_quality.query('indicator_value > 0'))) * 10}