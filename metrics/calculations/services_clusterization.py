import json
import pca
import os
import contextlib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

from scipy.cluster.hierarchy import fcluster, linkage
from .errors import TerritorialSelectError
from .base_method import BaseMethod


class ServicesClusterization(BaseMethod):
    def __init__(self, city_model):
        BaseMethod.__init__(self, city_model)
        super().validation("services_clusterization")
        self.services = self.city_model.Services.copy()

    def get_clusters_polygon(self, service_types, area_type = None, area_id = None, geojson = None, 
                            condition="distance", condition_value=4000, n_std = 2):

        services_select = self.services[self.services["service_code"].isin(service_types)]

        if area_type and area_id:
            services_select = self._get_territorial_select(area_type, area_id, services_select)[0]
        elif geojson:
            services_select = self._get_custom_polygon_select(geojson, self.city_crs, services_select)[0]

        if len(services_select) <= 1:
            raise TerritorialSelectError("services")

        services_select = self._get_service_cluster(services_select, condition, condition_value)

        # Find outliers of clusters and exclude it
        outlier = services_select.groupby("cluster", group_keys=False)["geometry"].apply(
            lambda x: self._find_dense_groups(x, n_std))
        services_outlier = services_select.loc[outlier]
        if any(~outlier):
            services_normal = services_select.loc[~outlier]

            if len(services_normal) > 0:
                cluster_service = services_normal.groupby(["cluster"], group_keys=True).apply(
                    lambda x: self._get_service_ratio(x))
                if isinstance(cluster_service, pd.Series):
                    cluster_service = cluster_service.unstack(level=1, fill_value=0)

                # Get MultiPoint from cluster Points and make polygon
                polygons_normal = services_normal.dissolve("cluster").convex_hull
                df_clusters_normal = pd.concat([cluster_service, polygons_normal.rename("geometry")], axis=1
                                                )
                cluster_normal = df_clusters_normal.index.max()
        else:
            df_clusters_normal = None

        # Select outliers 
        if any(outlier):
            services_outlier = services_select.loc[outlier].copy()

            # Reindex clusters
            clusters_outlier = cluster_normal + 1
            new_clusters = [c for c in range(clusters_outlier, clusters_outlier + len(services_outlier))]
            services_outlier.loc[:, "cluster"] = new_clusters
            cluster_service = services_outlier.groupby(["cluster"], group_keys=True).apply(lambda x: self._get_service_ratio(x))
            if isinstance(cluster_service, pd.Series):
                cluster_service = cluster_service.unstack(level=1, fill_value=0)
            df_clusters_outlier = cluster_service.join(services_outlier.set_index("cluster")["geometry"])
        else:
            services_outlier = None
            df_clusters_outlier = None

        df_clusters = pd.concat([df_clusters_normal, df_clusters_outlier]).fillna(0).set_geometry("geometry")
        df_clusters["geometry"] = df_clusters["geometry"].buffer(50, join_style=3)
        df_clusters = df_clusters.rename(columns={"index": "cluster_id"})

        services = pd.concat([services_normal, services_outlier]).set_crs(self.city_crs).to_crs(4326)
        df_clusters = df_clusters.set_crs(self.city_crs).to_crs(4326)

        return {"polygons": json.loads(df_clusters.to_json()), "services": json.loads(services.to_json())}

    @staticmethod
    def _get_service_cluster(services_select, condition, condition_value):
        services_coords = pd.DataFrame({"x": services_select.geometry.x, "y": services_select.geometry.y})
        clusterization = linkage(services_coords.to_numpy(), method="ward")
        services_select["cluster"] = fcluster(clusterization, t=condition_value, criterion=condition)
        return services_select

    @staticmethod
    def _find_dense_groups(loc, n_std):
        if len(loc) > 1:
            X = pd.DataFrame({"x": loc.x, "y": loc.y})
            X = X.to_numpy()

            # supress pca lib output
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                outlier = pca.spe_dmodx(X, n_std=n_std)[0]["y_bool_spe"]
            return pd.Series(data=outlier.values, index=loc.index)
        else:
            return pd.Series(data=True, index=loc.index)

    @staticmethod
    def _get_service_ratio(loc):
        all_services = loc["id"].count()
        services_count = loc.groupby("service_code")["id"].count()
        return (services_count / all_services).round(2)