# -*- coding: utf-8 -*-
"""Visualization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xgvU49GiN1GPgfce_peDUbtpta6L_ZLH
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('penguins.csv')
dataset.head()

dataset.isna().sum()

correctGender= ['female','male','female','female','female','female']
nullEntriesIndex = [8, 9, 10, 11, 47, 76]
mask = dataset['gender'].isnull()==True
nullEntries = dataset[mask]
dataset.drop(index=nullEntriesIndex,axis=0,inplace=True)
nullEntries

nullEntries['gender'] = correctGender
nullEntries

dataset = dataset.append(nullEntries)
dataset

dataset.isna().sum()

dataset['gender'] = [0 if gender =='female' else 1 for gender in dataset['gender']]
dataset['body_mass_g'] = dataset['body_mass_g']/1000
dataset['flipper_length_mm'] = dataset['flipper_length_mm']/10

adelie = dataset[dataset['species']=='Adelie']
chinstrap = dataset[dataset['species']=='Chinstrap']
gentoo = dataset[dataset['species']=='Gentoo']
features = dataset.columns.tolist()
features.remove('species')

def drawLinearityBetweenFeatures(featureOne,featureTwo, showAdelie,showChinstrap,showGentoo):
    legendList=[]
    if showAdelie==True:
        plt.scatter(adelie[featureOne],adelie[featureTwo], c="red")
        legendList.append('adelie')
    if showChinstrap==True:
        plt.scatter(chinstrap[featureOne],chinstrap[featureTwo], c= "black")
        legendList.append('chinstrap')
    if showGentoo==True:
        plt.scatter(gentoo[featureOne],gentoo[featureTwo], c= "blue")
        legendList.append('gentoo')
    plt.xlabel(featureOne)
    plt.ylabel(featureTwo)
    plt.legend(legendList,loc="upper right",fontsize=8.5)
    plt.show()

def visualize(showAdelie,showChinstrap,showGentoo):
    for feature in range(len(features)):
        for nextFeature in range (feature+1, len(features)):
            drawLinearityBetweenFeatures(features[feature],features[nextFeature],showAdelie,showChinstrap,showGentoo)

visualize(True,True,True)

visualize(True,True,False)

visualize(True,False,True)

visualize(False,True,True)