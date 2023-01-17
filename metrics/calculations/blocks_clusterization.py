import pandas as pd
import json
import numpy as np
import io

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from matplotlib import pyplot as plt
from .errors import SelectedValueError
from .base_method import BaseMethod


class BlocksClusterization(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("blocks_clusterization")
        self.services = self.city_model.Services.copy()
        self.blocks = self.city_model.Blocks.copy()

    def get_blocks(self, service_types, clusters_number=None, area_type=None, area_id=None, geojson=None):

        selected_services = self.services[self.services["service_code"].isin(service_types)]
        if len(selected_services) == 0: raise SelectedValueError("services", service_types, "service_code")
        clusterization, service_in_blocks = self._clusterize(selected_services)
        
        # If user doesn't specified the number of clusters, use default value.
        # The default value is determined with the rate of change in the distance between clusters
        if not clusters_number:
            clusters_number = self._get_clusters_number(clusterization)

        service_in_blocks["cluster_labels"] = fcluster(clusterization, t=int(clusters_number), criterion="maxclust")
        blocks = self.blocks.join(service_in_blocks, on="id")
        mean_services_number = service_in_blocks.groupby("cluster_labels").mean().round()
        mean_services_number = service_in_blocks[["cluster_labels"]].join(mean_services_number, on="cluster_labels")
        deviations_services_number = service_in_blocks - mean_services_number
        blocks = blocks.join(deviations_services_number, on="id", rsuffix="_deviation")
        if area_type and area_id:
            blocks = self._get_territorial_select(area_type, area_id, blocks)[0]
        elif geojson:
            blocks = self._get_custom_polygon_select(geojson, self.city_crs, blocks)[0]
        return json.loads(blocks.to_crs(4326).to_json())

    def get_dendrogram(self, service_types):
            
            selected_services = self.services[self.services["service_code"].isin(service_types)]
            if len(selected_services) == 0: raise SelectedValueError("services", service_types, "service_code")
            clusterization, service_in_blocks = self._clusterize(selected_services)

            img = io.BytesIO()
            plt.figure(figsize=(20, 10))
            plt.title("Dendrogram")
            plt.xlabel("Distance")
            plt.ylabel("Block clusters")
            dn = dendrogram(clusterization, p=7, truncate_mode="level")
            plt.savefig(img, format="png")
            plt.close()
            img.seek(0)

            return img

    def _clusterize(self, selected_services):

        service_in_blocks = selected_services.groupby(["block_id", "service_code"])["id"].count().unstack(fill_value=0)
        without_services = self.blocks["id"][~self.blocks["id"].isin(service_in_blocks.index)].values
        without_services = pd.DataFrame(columns=service_in_blocks.columns, index=without_services).fillna(0)
        service_in_blocks = pd.concat([without_services, service_in_blocks])
        clusterization = linkage(service_in_blocks, method="ward")
        return clusterization, service_in_blocks

    @staticmethod
    def _get_clusters_number(clusterization):

        distance = clusterization[-100:, 2]
        clusters = np.arange(1, len(distance) + 1)
        acceleration = np.diff(distance, 2)[::-1]
        series_acceleration = pd.Series(acceleration, index=clusters[:-2] + 1)

        # There are always more than two clusters
        series_acceleration = series_acceleration.iloc[1:]
        clusters_number = series_acceleration.idxmax()

        return clusters_number