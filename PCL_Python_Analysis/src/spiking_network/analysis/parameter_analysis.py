#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:26:24 2020

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns


#%% Correlation matrix

def correlation_matrix(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    np.fill_diagonal(corr.values, 0)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    sns.pairplot(df[["mean_sr", "std_sr"]])


#%%

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def PCA(df):
    # x = df.loc[:, features].values
    # y = df.loc[:, ["Mark"]].values
    # x = StandardScaler().fit_transform(x)
    
    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(
    #     data=principalComponents, columns=["principal component 1", "principal component 2"]
    # )
    # finalDf = pd.concat([principalDf, df[["Mark"]]], axis=1)
    
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel("Principal Component 1", fontsize=15)
    # ax.set_ylabel("Principal Component 2", fontsize=15)
    # ax.set_title("2 component PCA", fontsize=20)
    
    # targets = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # for target in targets:
    #     indicesToKeep = finalDf["Mark"] == target
    #     ax.scatter(
    #         finalDf.loc[indicesToKeep, "principal component 1"],
    #         finalDf.loc[indicesToKeep, "principal component 2"],
    #         s=50,
    #     )
    # ax.legend(targets)
    # ax.grid()
    pass


#%%

def scatter_mat(df):
    scatter_matrix(df, alpha=0.5, figsize=(6, 6), diagonal='kde')