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
import scipy.io as spio
from matplotlib import pyplot as plt
from dspca.utils import adjust_lightness
import h5py
from mat4py import loadmat


with h5py.File(r'C:\Users\lucid\OneDrive\Documents\GitHub\computeTopLevelScripts\Proj\dimensionRed\dsPCA\Data\Mouse1S2_activity.mat') as f:
    TAct=f['TAct'][()]
    dQ2=f['dQ'][()]
    Qch2=f['Qch'][()]
    sQ2=f['sQ'][()]
    f.close()
# For converting to a NumPy array
with np.load(r'C:\Users\lucid\OneDrive\Documents\GitHub\dspca\data\data.npz') as data:
    dQ = data['dQ']  # Target
    Qch = data['Qch']  # Target
    sQ = data['sQ']  # Target
    activity = data['activity']  # Neural population activity (Trial X Time X Cell)
    
    
# dQ2 = matDataImport['dQ']  # Target
# Qch2 = matDataImport['Qch']  # Target
# sQ2 = matDataImport['sQ']  # Target
# activity2 = matDataImport['TAct']  # Neural population activity (Trial X Time X Cell)

TAct2=TAct.transpose()    
targets = np.vstack((dQ2, Qch2, sQ2)).T    # Target task-related variables
time_range = np.arange(10, 15)  # Time range used to identify subspaces
activity_mean = np.mean(TAct2[:, time_range, :], axis=1)  # Temporally averaged neural population activity (Trial X Cell)


projection_target_subspace, projection_targetfree_subspace, ax_targets, ax_targetfree, \
target_subspace_signal, targetfree_subspace_signal, target_subspace_var, targetfree_subspace_var, total_var, dot_target_ax\
    = dsPCA(data=activity_mean, targets=targets)
