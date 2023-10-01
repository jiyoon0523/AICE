#1.데이터 로드
import pandas as pd
import numpy as np

df = pd.read_csv('TrainData.csv',delimiter=',')
df
df.info()
df.describe()

#2.데이터 탐색/정제
##2-1.중복 데이터 제거
df.info()
df= df.drop_duplicates()
df.info()

##2-2.텍스트, 범주형 데이터 처리
df_ex = pd.DataFrame({'name': ['Alice','Bob','Charlie','Dave','Ellen','Frank'],
                   'age': [24,42,18,68,24,30],
                   'state': ['NY','CA','CA','TX','CA','NY'],
                   'point': [64,24,70,70,88,57]}
                  )
print(df_ex)

df_ex['state'].replace({'CA':'California','NY':'NewYork'}, inplace= True)
print(df_ex)

df['Result_v1'].unique()
df['Result_v1'].replace({'benign':1, 'malicious':-1}, inplace= True)
df['Result_v1'].unique()

#3.결측치 제거
df.info()
df = df.dropna(axis= 0)
df.info()

#4.데이터 탐색을 통해 불필요한 칼럼 제거
##4-1. corr 사용
df.corr()
top_10= df.corr()['Result_v1'].sort_values(ascending= False)
top_10

##4-2.scatter 그래프 사용
import matplotlib.pyplot as plt

df['color']= df['Result_v1'].map({1:'blue', -1:'red'})
df

y_list= df.columns

for i in range(0, len(y_list)):
    df.plot(kind= 'scatter', x= 'Result_v1', y= y_list[i], s= 30, c= df['color'])
    plt.title("Scatter Bengin Malicious", fontsize= 20)
    plt.xlabel('Result_v1')
    plt.ylabel(y_list[i])
    plt.show()

##4-3.불필요한 칼럼 제거
df.drop(columns= ['url_chinese_present', "html_num_tags('applet')"], inplace= True)
df.info()

#5.train set과 test set 분리
from sklearn.model_selection import train_test_split

X= df.iloc[:, 0:len(df.columns)-1].values
y= df.iloc[:, len(df.columns)-1].values

train_x, test_x, train_y, test_y= train_test_split(X, y, test_size= 0.3, random_state=2021)
train_x.shape, test_x.shape, train_y.shape, test_y.shape