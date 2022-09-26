# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:22:58 2022

@author: lucid
"""

import pymc3 as pm
import pandas as pd
data = pd.read_csv(   'https://raw.githubusercontent.com/Garve/datasets/3b6b1e6fadc04e2444905db0a0b2ed222daeaa28/rolling_data.csv',
    index_col='t'
)


with pm.Model() as rolling_linear_model:
    # noise parameters
    sigma_slope = pm.Exponential('sigma_slope', lam=1)
    sigma_intercept = pm.Exponential('sigma_intercept', lam=1)
    sigma = pm.Exponential('sigma', lam=1)
    
    # Gaussian random walks
    slopes = pm.GaussianRandomWalk(
        'slopes',
        sigma=sigma_slope,
        shape=len(data)
    )
    intercepts = pm.GaussianRandomWalk(
        'intercepts',
        sigma=sigma_intercept,
        shape=len(data)
    )
    
    # putting it together
    y = pm.Normal(
        'y',
        slopes*data['x'] + intercepts,
        sigma,
        observed=data['y']
    )
    
    rolling_linear_trace = pm.sample(return_inferencedata=True)