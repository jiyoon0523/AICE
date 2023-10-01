#1.데이터 불러오기
import pandas as pd
import numpy as np

df= pd.read_csv("onenavi_train.csv", sep= "|")
df_eval= pd.read_csv("onenavi_evaluation.csv", sep= "|")
df_total= pd.concat([df, df_eval], ignore_index= True)
df_total

df_pnu=pd.read_csv("onenavi_pnu.csv",sep="|")
# df_signal = pd.read_csv("onenavi_signal.csv",sep="|")
df_total= pd.merge(df_total, df_pnu, on= "RID")
df_total

#2.이상치/결측치 처리
##2-1.결측치 처리
df_total.isnull().sum()

sample= pd.DataFrame(
    {
        'col1': [50,70,np.nan,55],
        'col2': [22,50,66,np.nan]
    })
sample

# sample.isnull().sum()
# sample.dropna()
sample.fillna(method= "pad")

##2-2.이상치 처리
###2-2-1.pairplot에서 보이는 outlier
# !pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df_total)
plt.show()

df_total[df_total['ET']>25000]
df_total= df_total[df_total['ET']<=25000]
df_total

sns.pairplot(df_total)
plt.show()

###2-2-2.변수로부터 새로운 데이터를 도출했을 떄 보이는 outlier- 시속
df_total['PerHour']= df_total['A_DISTANCE']/df_total['ET'] *(3600/1000)
df_total.describe()

len(df_total[df_total['PerHour']>=130])
df_total= df_total[df_total['PerHour']<130]
df_total

###2-2-3.관심범위 밖의 데이터 제거
df_total['level1_pnu'].unique()
df_total= df_total[(df_total['level1_pnu']=='경기도')|(df_total['level1_pnu']=='서울특별시')|(df_total['level1_pnu']=='인천광역시')]
df_total= df_total.reset_index(drop=True)
df_total

#3.더미변수 생성
# !pip install tqdm
import datetime
from dateutil.parser import parse
from tqdm import tqdm

weekday_list=[]
hour_list=[]
day_list=[]

for w in tqdm(df_total['TIME_DEPARTUREDATE']):
    parse_data_w= parse(w)
    weekday_list.append(parse_data_w.weekday())
    hour_list.append(parse_data_w.hour)
    day_list.append(parse_data_w.day)
    
df_total['WEEKDAY']= weekday_list
df_total['HOUR']= hour_list
df_total['DAY']= day_list

df_total

dummy_fields= ['WEEKDAY', 'HOUR', 'DAY', 'level1_pnu']

for dummy in dummy_fields:
    dummies= pd.get_dummies(df_total[dummy], prefix= dummy, drop_first= False)
    df_total= pd.concat([df_total, dummies], axis=1)
    
df_total= df_total.drop(dummy_fields, axis=1)
df_total

#4.데이터 스케일링
data_day= df_total['DAY']

train_data= df_total.drop(['RID', 'TIME_DEPARTUREDATE', 'TIME_ARRIVEDATE', 'ET', 'ETAA', 'DAY', 'level1_pnu', 'level2_pnu'], axis=1)
columnNames= train_data.columns                        
train_data

from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler(feature_range=(0,1))
feature= pd.DataFrame(scaler.fit_transform(train_data))
feature.columns= columnNames
feature

#5.저장 
feature['DAY']= data_day 

train_feature= feature[feature['DAY']<=24]
train_feature= train_feature.drop(['DAY'], axis=1)

eval_feature= feature[feature['DAY']>24]
eval_feature= eval_feature.drop(['DAY'], axis=1)

len(feature), len(train_feature), len(eval_feature)

train_target= df_total[df_total['DAY']<=24]['ET']
train_target

train_feature.to_csv("onenavi_train_feature.csv", index= False, sep= "|")
eval_feature.to_csv("onenavi_eval_feature.csv", index= False, sep= "|")
train_target.to_csv("onenavi_train_target.csv", index= False, sep="|")

