# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:22:56 2022

@author: lucid
"""

from dspca.dspca import dsPCA
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import csv
from matplotlib import pyplot as plt
from dspca.utils import adjust_lightness

file = open("Salary_Data.csv")
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows = []
for row in csvreader:
    rows.append(row)
print(rows)
file.close()


with np.load(r'C:\Users\lucid\OneDrive\Documents\GitHub\dspca\data\data.npz') as data:
    dQ = data['dQ']  # Target
    Qch = data['Qch']  # Target
    sQ = data['sQ']  # Target
    activity = data['activity']  # Neural population activity (Trial X Time X Cell)
    
    targets = np.vstack((dQ, Qch, sQ)).T    # Target task-related variables
time_range = np.arange(10, 15)  # Time range used to identify subspaces
activity_mean = np.mean(activity[:, time_range, :], axis=1)  # Temporally averaged neural population activity (Trial X Cell)

projection_target_subspace, projection_targetfree_subspace, ax_targets, ax_targetfree, \
target_subspace_signal, targetfree_subspace_signal, target_subspace_var, targetfree_subspace_var, total_var, dot_target_ax\
    = dsPCA(data=activity_mean, targets=targets)
