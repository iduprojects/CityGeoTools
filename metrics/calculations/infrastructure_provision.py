import geopandas as gpd
import pandas as pd
import json

from .base_method import BaseMethod
from .provision import CityProvision
from .diversity import Diversity

class InfrastructureProvisionIndex(BaseMethod):
    '''
    >>> InfrastructureProvisionIndex(city_model).get_ip_index()
    returns infrastructure provision index and raw data for it
    metric calculates provision and diversity for several groups of services
    and returns rank of infrastructure provision for each city block (from 1 to 5, and 0 is for missing data)
    '''
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        self.blocks = city_model.Blocks.copy()
        self.greenery = city_model.RecreationalAreas.copy()
    
    @staticmethod
    def _ind_ranking(data_series):
        return pd.to_numeric(pd.cut(data_series, 5, labels=[1, 2, 3, 4, 5], right=False))

    def _ipi1(self):
        '''
        calculates school provision
        '''
        local_blocks = self.blocks.copy()
        Provisions_class = CityProvision(self.city_model,
                                    service_types = ['schools'],
                                    valuation_type = "normative",
                                    year = 2022, 
                                    user_changes_buildings = None,
                                    user_changes_services = None,
                                    user_provisions = None,
                                    user_selection_zone = None,
                                    service_impotancy = None)
        local_provision = Provisions_class.get_provisions().buildings

        local_blocks = local_blocks.join(local_provision[['block_id', 'schools_provison_value']].groupby('block_id').mean(), on='id')
        local_blocks.rename(columns={'schools_provison_value':'IPI_data'}, inplace=True)
        local_blocks['IPI'] = self._ind_ranking(local_blocks['IPI_data'])
        print('Indicator 1 done')
        return local_blocks['IPI'], local_blocks['IPI_data']

    def _ipi2(self):
        '''
        calculates kindergarten provision
        '''
        local_blocks = self.blocks.copy()
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
        local_blocks.rename(columns={'kindergartens_provison_value':'IPI_data'}, inplace=True)
        local_blocks['IPI'] = self._ind_ranking(local_blocks['IPI_data'])
        print('Indicator 2 done')
        return local_blocks['IPI'], local_blocks['IPI_data']

    def _ipi3(self):
        '''
        calculates number of policlinics in 1000 meters isochrone
        '''
        local_blocks = self.blocks.copy()
        objects = 'policlinics'

        Div_class = Diversity(self.city_model)
        local_diversity = Div_class.get_diversity_v2(objects, return_jsons=False, valuation_type='custom',\
             travel_type='walk', limit_value=1000)
        local_blocks['IPI_data'] = local_diversity.blocks.diversity
        local_blocks['IPI'] = self._ind_ranking(local_blocks['IPI_data'])
        print('Indicator 3 done')
        return local_blocks['IPI'], local_blocks['IPI_data']

    def _ipi4(self):
        '''
        calculates number of sport objects in 1000 meters isochrone
        '''
        local_blocks = self.blocks.copy()
        objects = ['swimming_pools', 'aquaparks', 'fitness_clubs', 'sport_centers', 'sport_clubs',
        'stadiums', 'sport_section']

        Div_class = Diversity(self.city_model)
        local_diversity = Div_class.get_diversity_v2(objects, return_jsons=False, valuation_type='custom',\
             travel_type='walk', limit_value=1000)
        local_blocks['IPI_data'] = local_diversity.blocks.diversity
        local_blocks['IPI'] = self._ind_ranking(local_blocks['IPI_data'])
        print('Indicator 4 done')
        return local_blocks['IPI'], local_blocks['IPI_data']

    def _ipi5(self):
        '''
        calculates number of cultural and recreational services in 15 minutes isochrone
        '''
        local_blocks = self.blocks.copy()
        objects = ['art_spaces', 'zoos', 'libraries', 'theaters', 'museums', 'cinemas', 'shopping_centers', 'bowlings',
        'clubs', 'child_teenager_club', 'culture_house', 'quest', 'circus', 'art_gallery', 'music_school', 'amusement_park']

        Div_class = Diversity(self.city_model)
        local_diversity = Div_class.get_diversity_v2(objects, return_jsons=False, valuation_type='custom',\
             travel_type='public_transport', limit_value=15)
        local_blocks['IPI_data'] = local_diversity.blocks.diversity
        local_blocks['IPI'] = self._ind_ranking(local_blocks['IPI_data'])
        print('Indicator 5 done')
        return local_blocks['IPI'], local_blocks['IPI_data']

    def _ipi6(self):
        '''
        calculates number of commercial services in 15 minutes isochrone
        '''
        local_blocks = self.blocks.copy()
        objects = ['clothing_repairings', 'photo_studios', 'watch_repairings', 'bookmaker_offices', 'rituals', 'services', 'notary',
        'recruitment_agency', 'insurance', 'property_appraisal', 'legal_services', 'charitable_foundations', 'copy_center', 'real_estate_agency',
        'wedding_agency', 'tourist_agency']
        #need to add food, goods and mean result
        Div_class = Diversity(self.city_model)
        local_diversity = Div_class.get_diversity_v2(objects, return_jsons=False, valuation_type='custom',\
             travel_type='public_transport', limit_value=15)
        local_blocks['IPI_data'] = local_diversity.blocks.diversity
        local_blocks['IPI'] = self._ind_ranking(local_blocks['IPI_data'])
        print('Indicator 6 done')
        return local_blocks['IPI'], local_blocks['IPI_data']

    def _ipi8(self):
        '''
        calculates share of greenery in blocks
        '''
        local_blocks = self.blocks.copy()
        local_greenery = self.greenery.copy()
        local_greenery = local_greenery.explode(ignore_index=True)
        local_greenery = local_greenery[local_greenery.geometry.type =="Polygon"]
        local_blocks['area'] = local_blocks.area
        greenery_in_blocks = gpd.overlay(local_blocks, local_greenery, how='intersection')
        greenery_in_blocks['green_area'] = greenery_in_blocks.area
        share_of_green = greenery_in_blocks[['block_id', 'area', 'green_area']].groupby('block_id').sum()
        share_of_green['IPI_data'] = share_of_green['green_area'] / share_of_green['area']
        local_blocks = local_blocks.join(share_of_green['IPI_data'], on='id')

        local_blocks['IPI'] = self._ind_ranking(local_blocks['IPI_data'])
        print('Indicator 8 done')
        return local_blocks['IPI'], local_blocks['IPI_data']

    def get_ip_index(self):

        ipi_blocks = self.blocks.copy().to_crs(4326)

        ipi_blocks['ipi1'], ipi_blocks['data_ipi1'] = self._ipi1()
        ipi_blocks['ipi2'], ipi_blocks['data_ipi2'] = self._ipi2()
        ipi_blocks['ipi3'], ipi_blocks['data_ipi3'] = self._ipi3()
        ipi_blocks['ipi4'], ipi_blocks['data_ipi4'] = self._ipi4()
        ipi_blocks['ipi5'], ipi_blocks['data_ipi5'] = self._ipi5()
        ipi_blocks['ipi6'], ipi_blocks['data_ipi6'] = self._ipi6()
        #ipi_blocks['ipi7'], ipi_blocks['data_ipi7'] = self._ipi7() #crosses with ind 4, ind5 --> waiting for public spaces as distinct objects
        ipi_blocks['ipi8'], ipi_blocks['data_ipi8'] = self._ipi8()

        ipi_blocks['infrastructure_provision_index'] = ipi_blocks.filter(regex='^ipi.*').mean(axis=1).round(0)
        ipi_blocks = ipi_blocks.fillna(0)
        
        return json.loads(ipi_blocks.to_json())