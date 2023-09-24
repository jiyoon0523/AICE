import pandas as pd
import numpy as np

#1. 데이터 불러오기
##Q1.
df= pd.read_csv("onenavi_train.csv")
df.shape

##Q2.
df_eval= pd.read_csv("onenavi_evaluation.csv")
df_total= pd.merge(df, )
df_total.shape

#2. 추가변수 생성
##Q3.
df_region= pd.read_csv("")
df_traffic_light= pd.read.csv("")
df_total= pd.merge(df_total, df_region, df_traffic_light)

#3. 데이터 분석
##Q4.
df_weather= pd.read_csv("onenavi_weather.csv")
df_total_temp= pd.merge(df_total, df_weather)

##Q5.
import seaborn as sns

##Q6.
corr= df_total.corr()
sns.heatmap(corr, annot= True)

