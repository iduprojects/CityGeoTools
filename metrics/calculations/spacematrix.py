import pandas as pd
import json
import numpy as np
import pandas.core.indexes as pd_index

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .errors import SelectedValueError
from .base_method import BaseMethod


class Spacematrix(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("spacematrix")
        self.buildings = self.city_model.Buildings.copy()
        self.blocks = self.city_model.Blocks.copy().set_index("id")

    def get_morphotypes(self, clusters_number=11, area_type=None, area_id=None, geojson=None):

        buildings, blocks = self._simple_preprocess_data(self.buildings, self.blocks)
        blocks = self._calculate_block_indices(buildings, blocks)

        blocks = self.get_spacematrix_morph_types(blocks, clusters_number)

        blocks = self._get_strelka_morph_types(blocks)

        if area_type and area_id:
            if area_type == "block":
                try: 
                    blocks = blocks.loc[[area_id]]
                except:
                    raise SelectedValueError("build-up block", "area_id", "id")
            else:
                blocks = self._get_territorial_select(area_type, area_id, blocks)[0]
        elif geojson:
            blocks = self._get_custom_polygon_select(geojson, self.city_crs, blocks)[0]

        return json.loads(blocks.reset_index().to_crs(4326).to_json())

    @staticmethod
    def _simple_preprocess_data(buildings, blocks):

            # temporary filters. since there are a few bugs in buildings table from DB
        buildings = buildings[buildings["block_id"].notna()]
        buildings = buildings[buildings["storeys_count"].notna()]
        buildings["is_living"] = buildings["is_living"].fillna(False)
        buildings["building_area"] = buildings["basement_area"] * buildings["storeys_count"]
        bad_living_area = buildings[buildings["living_area"] > buildings["building_area"]].index
        buildings.loc[bad_living_area, "living_area"] = None
        living_grouper = buildings.groupby(["is_living"])
        living_area = living_grouper.apply(
            lambda x: x.living_area.fillna(x.building_area * 0.8) if x.name else x.living_area.fillna(0)
            )
        if type(living_area.index) == pd_index.multi.MultiIndex:
            buildings["living_area"] = living_area.droplevel(0).round(2)
        else:
            buildings["living_area"] = living_area.values[0].round(2)

        return buildings, blocks

    @staticmethod
    def _calculate_block_indices(buildings, blocks):
        sum_grouper = buildings.groupby(["block_id"]).sum(numeric_only=True)
        blocks["FSI"] = sum_grouper["building_area"] / blocks["area"]
        blocks["GSI"] = sum_grouper["basement_area"] / blocks["area"]
        blocks["MXI"] = (sum_grouper["living_area"] / sum_grouper["building_area"]).round(2)
        blocks["L"] = (blocks["FSI"] / blocks["GSI"]).round()
        blocks["OSR"] = ((1 - blocks["GSI"]) / blocks["FSI"]).round(2)
        blocks[["FSI", "GSI"]] = blocks[["FSI", "GSI"]].round(2)
        
        return blocks

    @staticmethod
    def _name_spacematrix_morph_types(cluster):
        ranges = [[0, 3, 6, 10, 17], 
                  [0, 1, 2], 
                  [0, 0.22, 0.55]]
        labels = [["Малоэтажный", "Среднеэтажный", "Повышенной этажности", "Многоэтажный", "Высотный"],
                  [" низкоплотный", "", " плотный"], 
                  [" нежилой", " смешанный", " жилой"]]
        cluster_name = []
        for ind in range(len(cluster)):
            cluster_name.append(
                labels[ind][[i for i in range(len(ranges[ind])) if cluster.iloc[ind] >= ranges[ind][i]][-1]]
                )
        return "".join(cluster_name)

    def get_spacematrix_morph_types(self, blocks, clusters_number):
        # blocks with OSR >=10 considered as unbuilt blocks
        X = blocks[blocks["OSR"] < 10][['FSI', 'L', 'MXI']].dropna()
        scaler = StandardScaler()
        X_scaler = pd.DataFrame(scaler.fit_transform(X))
        kmeans = KMeans(n_clusters=clusters_number, random_state=42).fit(X_scaler)
        X["spacematrix_cluster"] = kmeans.labels_
        blocks = blocks.join(X["spacematrix_cluster"])
        cluster_grouper = blocks.groupby(["spacematrix_cluster"]).median(numeric_only=True)
        named_clusters = cluster_grouper[["L", "FSI", "MXI"]].apply(
            lambda x: self._name_spacematrix_morph_types(x), axis=1)
        blocks = blocks.join(named_clusters.rename("spacematrix_morphotype"), on="spacematrix_cluster")
        
        return blocks
        
    @staticmethod
    def _get_strelka_morph_types(blocks):

        storeys = [blocks['L'].between(0,3), blocks['L'].between(4,8), (blocks['L']>=9)]
        labels = ['Малоэтажная застройка', 'Среднеэтажная застройка', 'Многоэтажная застройка']
        blocks['strelka_morphotype'] = np.select(storeys, labels, default='Другое')

        mxis = [(blocks["strelka_morphotype"] == 'Малоэтажная застройка') & (blocks['MXI']<0.05),
                (blocks["strelka_morphotype"] == 'Среднеэтажная застройка') & (blocks['MXI']<0.2),
                (blocks["strelka_morphotype"] == 'Многоэтажная застройка') & (blocks['MXI']<0.1)]
        labels = ['Малоэтажная нежилая застройка', 'Среднеэтажная нежилая застройка', 'Многоэтажная нежилая застройка']
        blocks['strelka_morphotype'] = np.select(mxis, labels, default = blocks["strelka_morphotype"])

        conds = [(blocks['strelka_morphotype'] == 'Малоэтажная застройка') & ((blocks['FSI']*10)<=1),
                 (blocks['strelka_morphotype'] == 'Малоэтажная застройка') & ((blocks['FSI']*10)>1),
                 (blocks['strelka_morphotype'] == 'Среднеэтажная застройка') & ((blocks['FSI']*10)<=8) & (blocks['MXI']<0.45),
                 (blocks['strelka_morphotype'] == 'Среднеэтажная застройка') & ((blocks['FSI']*10)>8) & (blocks['MXI']<0.45),
                 (blocks['strelka_morphotype'] == 'Среднеэтажная застройка') & ((blocks['FSI']*10)>15) & (blocks['MXI']>=0.6),
                 (blocks['strelka_morphotype'] == 'Многоэтажная застройка') & ((blocks['FSI']*10)<=15),
                 (blocks['strelka_morphotype'] == 'Многоэтажная застройка') & ((blocks['FSI']*10)>15)]
        labels = ['Индивидуальная жилая застройка',
                  'Малоэтажная модель застройки',
                  'Среднеэтажная микрорайонная застройка',
                  'Среднеэтажная квартальная застройка',
                  'Центральная модель застройки',
                  'Многоэтажная советская микрорайонная застройка',
                  'Многоэтажная соверменная микрорайонная застройка']
        blocks['strelka_morphotype'] = np.select(conds, labels, default=blocks["strelka_morphotype"])

        return blocks
