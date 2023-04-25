from typing import Any, Optional
from .base_method import BaseMethod

import pandas as pd
import geopandas as gpd
import numpy as np
 
from typing import Iterable

class PopsResetelment:

    
    def default_distr(distributed_value: int, 
                      probabilities_distribution: Iterable,
                      probabilities_names: Iterable):
        
        distributed_value = int(distributed_value)

        rng = np.random.default_rng(seed = 0)
        r = rng.choice(probabilities_names,
                       distributed_value, 
                       replace = True, 
                       p = probabilities_distribution)
        r = np.unique(r, return_counts = True)
        t_ = dict(zip(r[0],  r[1]))
        for g in probabilities_names:
            if g in t_.keys():
                pass
            else: 
                t_.update({g: 0})
        return t_