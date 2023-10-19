#1.모델 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import GradientBoostingRegressor as grb
from xgboost import XGBRegressor as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

import pickle
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

##1.2 평가 데이터 
df_evaluation = pd.read_csv("onenavi_evaluation_new.csv",sep="|")
df_evaluation_feature = pd.read_csv("onenavi_eval_feature.csv",sep="|")
len(df_evaluation_feature)

##1.3 모델 불러오기 
model_rslt = []

for i in range(4):
    model_rslt.append(joblib.load("{}_model.pkl".format(i)))
model_rslt.append(keras.models.load_model("DeeplearningModel.h5"))
model_rslt

##1.4 모델 결과 표 만들기
e1_list = ['ETA1', 'ETA2', 'ETA3', 'ETA4', 'ETA5']
e2_list = ['ETAA1', 'ETAA2', 'ETAA3', 'ETAA4', 'ETAA5']

for e1, e2, model in zip(e1_list, e2_list, model_rslt):
    df_evaluation[e1] = model.predict(df_evaluation_feature)
    df_evaluation.loc[(df_evaluation[e1] < 0), e1] = 0
    etaa = (1-(abs(df_evaluation['ET']-df_evaluation[e1])/df_evaluation['ET']))*100.0
    df_evaluation[e2] = etaa
    df_evaluation.loc[(df_evaluation[e2] < 0), e2] = 0

etaa = ['ETAA', 'ETAA1', 'ETAA2', 'ETAA3', 'ETAA4', 'ETAA5']
alg = ['DATA', 'ML-LG', 'ML-RFR', 'ML-GBR', 'XBR', 'Deep']

print('+-------------------------------------------------------+')
print('|   ALG    | Mean(%) |    STD    |  MIN(%)  |  MAX(%)   |')
print('+----------+---------+-----------+----------+-----------+')
for i, e in zip(range(len(alg)), etaa):
    eMean = df_evaluation[e].mean()
    eStd = df_evaluation[e].std()
    eMin = df_evaluation[e].min()
    eMax = df_evaluation[e].max()
    print('|  {:6s}  |   {:3.1f}  |   {:05.1f}   |   {:4.1f}   |  {:7.1f}  | '.format(alg[i], eMean, eStd, eMin, eMax))
print('+----------+---------+-----------+----------+-----------+\n\n')