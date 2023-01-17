import pandas as pd
import json

from itertools import product
from .base_method import BaseMethod


class CollocationMatrix(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("collocation_matrix")
        self.services = self.city_model.Services.copy()

    def get_collocation_matrix(self):
        services = self.services.dropna().reset_index(drop=True)[['service_code','block_id']].sort_values('service_code')
        types_of_services = [
            "dentists", "pharmacies", "markets", "conveniences", "supermarkets", "art_spaces", "zoos", "libraries",
            "theaters", "museums", "cinemas", "bars", "bakeries", "cafes", "restaurants", "fastfoods", "saunas",
            "sportgrounds", "swimming_pools", "banks", "atms", "shopping_centers", "aquaparks", "fitness_clubs",
            "sport_centers", "sport_clubs", "stadiums", "beauty_salons", "spas", "metro_stations", "hardware_stores",
            "instrument_stores", "electronic_stores", "clothing_stores", "tobacco_stores", "sporting_stores",
            "jewelry_stores", "flower_stores", "pawnshops", "recreational_areas", "embankments", "souvenir_shops",
            "bowlings", "stops", "clubs", "microloan", "child_teenager_club", "sport_section", "culture_house", "quest",
            "circus", "child_game_room", "child_goods", "art_gallery", "book_store", "music_school", "art_goods",
            "mother_child_room", "holiday_goods", "toy_store", "beach", "amusement_park"
            ]
        services = services[services['service_code'].isin(types_of_services)]
        services['count'] = 0
        collocation_matrix = self._get_numerator(services) / self._get_denominator(services)

        return json.loads(collocation_matrix.to_json())

    @staticmethod
    def _get_numerator(services):
        services_numerator = services.pivot_table(index='block_id', columns='service_code', values='count')

        pairs_services_numerator = [(a, b) for idx, a in enumerate(services_numerator) for b in services_numerator[idx + 1:]]
        pairs_services_numerator = dict.fromkeys(pairs_services_numerator, 0)

        res_numerator = {}
        n_col = len(services_numerator.columns)
        for i in range(n_col):
            for j in range(i + 1, n_col):
                col1 = services_numerator.columns[i]
                col2 = services_numerator.columns[j]
                res_numerator[col1, col2] = sum(services_numerator[col1] == services_numerator[col2])
                res_numerator[col2, col1] = sum(services_numerator[col2] == services_numerator[col1])
        pairs_services_numerator.update(res_numerator)

        numerator = pd.Series(pairs_services_numerator).reset_index(drop=False).set_index(['level_0','level_1']).rename(columns={0:'count'})
        numerator = numerator.pivot_table(index='level_0', columns='level_1', values='count')

        return numerator

    def _get_denominator(self, services):
        count_type_block = services.groupby('service_code')['block_id'].nunique().reset_index(drop=False)

        pairs_services_denominator = []
        for i in product(count_type_block['service_code'], repeat=2):
            pairs_services_denominator.append(i)

        types_blocks_sum = []
        for i,j in pairs_services_denominator:
            if [i] ==[j]:
                    types_blocks_sum.append(0)
            else:
                    num1 = count_type_block.loc[count_type_block['service_code'] == i, 'block_id'].iloc[0]
                    num2 = count_type_block.loc[count_type_block['service_code'] == j, 'block_id'].iloc[0]
                    types_blocks_sum.append(num1+num2)

        res_denominator = {}
        for row in range(len(pairs_services_denominator)):
            res_denominator[pairs_services_denominator[row]] = types_blocks_sum[row]

        sum_res_denominator = pd.Series(res_denominator).reset_index(drop=False).set_index(['level_0','level_1']).rename(columns={0:'count'})
        sum_res_denominator = sum_res_denominator.pivot_table(index='level_0', columns='level_1', values='count')

        denominator = sum_res_denominator - self._get_numerator(services)

        return denominator