import geopandas as gpd
import pandas as pd
import json
import numpy as np

from .base_method import BaseMethod


class Masterplan(BaseMethod):

    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("masterplan")
        self.buildings = self.city_model.Buildings.copy()

    def get_masterplan(self, polygon: json, add_building: json, delet_building: list[int]) -> json:

        """The function calculates the indicators of the master plan for a certain territory.

        :param polygon: the territory within which indicators will be calculated in GeoJSON format.
        :param add_building: the building that will be added to the territory in GeoJSON format.
        :param delet_building: the building that will be deleted from the territory in List format.

        city_model = CityInformationModel(city_name="saint-petersburg", city_crs=32636)
        
        :example: Masterplan(city_model).get_masterplan(polygon, add_building, delet_building=[1, 2, 3])
        
        :return: dictionary with the name of the indicator and its value in JSON format.
        """

        polygon = gpd.GeoDataFrame.from_features([polygon]).set_crs(4326).to_crs(self.city_model.city_crs)
        land_with_buildings = gpd.sjoin(self.buildings, polygon, how='inner')

        if add_building is not None:
            add_building = gpd.GeoDataFrame.from_features(add_building).set_crs(4326).to_crs(self.city_model.city_crs)
            land_with_buildings = land_with_buildings.append(add_building)

        if delet_building is not None:
            delet_building = pd.DataFrame(delet_building)
            delet_building.columns = ['functional_object_id']
            land_with_buildings = land_with_buildings[~land_with_buildings['functional_object_id'].isin(
                delet_building['functional_object_id'])]
            
        land_with_buildings_living = land_with_buildings[land_with_buildings['is_living'] == True]
        
        hectare = 10000
        living = 80
        commerce = 20                             
        
        land_area =  polygon.area / hectare
        land_area = land_area.squeeze()
        land_area =  np.around(land_area, decimals = 2)
          
        buildings_area = land_with_buildings['basement_area'].sum()

        dev_land_procent = ((buildings_area / hectare) / land_area) * 100
        dev_land_procent = np.around(dev_land_procent, decimals = 2)
      
        dev_land_area = land_with_buildings['basement_area'] * land_with_buildings['storeys_count']
        dev_land_area = dev_land_area.sum() / hectare
        dev_land_area = np.around(dev_land_area, decimals = 2)
    
        dev_land_density = dev_land_area / land_area
        dev_land_density = np.around(dev_land_density, decimals = 2)

        land_living_area = land_with_buildings_living['basement_area'] * land_with_buildings_living['storeys_count']
        land_living_area = ((land_living_area.sum() / hectare) / 100 * living)
        land_living_area = np.around(land_living_area, decimals = 2)
   
        dev_living_density = land_living_area / land_area
        dev_living_density = np.around(dev_living_density, decimals = 2)

        population =  land_with_buildings['population'].sum()
        population = population.astype(int)
            
        population_density = population / land_area.squeeze()
        population_density = np.around(population_density, decimals = 2)

        living_area_provision = (land_living_area * hectare) / population
        living_area_provision = np.around(living_area_provision, decimals = 2)
        
        land_business_area = ((land_living_area / living) * commerce) 
        land_business_area = np.around(land_business_area, decimals = 2)
        
        building_height_mode = land_with_buildings['storeys_count'].mode().squeeze()
        building_height_mode = building_height_mode.astype(int)
            
        data = [land_area, dev_land_procent, dev_land_area, dev_land_density, land_living_area,
                    dev_living_density, population, population_density, living_area_provision, 
                    land_business_area, building_height_mode]   
        index = ['land_area', 'dev_land_procent',
                'dev_land_area', 'dev_land_density', 'land_living_area', 
                'dev_living_density', 'population', 
                'population_density', 'living_area_provision', 
                'land_business_area', 'building_height_mode']
        df_indicators = pd.Series(data, index=index)

        return json.loads(df_indicators.to_json())