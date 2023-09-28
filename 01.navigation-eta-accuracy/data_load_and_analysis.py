#1.데이터 불러오기
##1-1.라이브러리 import
import pandas as pd
import numpy as np

##1-2.데이터 프레임 변수로 저장
df= pd.read_csv("onenavi_train.csv", sep="|")
df

##1-3.학습 데이터와 평가 데이터 합치기
df_train= pd.read_csv("onenavi_train.csv", sep="|")
df_eval= pd.read_csv("onenavi_evaluation.csv", sep="|")
df_train.info()
df_eval.info()
df_total= pd.concat([df_train, df_eval], ignore_index=False)
# df_total= pd.merge(df_train, df_eval, on= "RID")
df_total

#2.추가변수 생성
df_pnu = pd.read_csv("onenavi_pnu.csv",sep="|") 
df_signal = pd.read_csv("onenavi_signal.csv",sep="|") 

df_total=pd.merge(df_total, df_pnu, on=["RID"])
df_total=pd.merge(df_total, df_signal, on="RID")
df_total

#3.데이터 분석
##3-1.라이브러리 import
# !pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

##3-2.Seaborn을 활용한 데이터 시각화
#CountChart
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()

font_list= [font.name for font in fm.fontManager.ttflist]
font_list

sns.set(font="NanumGothicCoding",
        rc= {"axes.unicode_minus":False},
        style='darkgrid')
ax= sns.countplot(x=df_total['level1_pnu_x'], palette= "RdBu")

#DistChart
sns.displot(x= df_total['signaltype'])
plt.show()

#BoxPlot
sns.boxplot(x= df_total['level1_pnu_x'], y= df_total['A_DISTANCE'], data= df_total, palette="RdBu")
plt.show()

#Heatmap
uniform_data= np.random.rand(10,12)
sns.heatmap(uniform_data)
plt.show()

#PairPlot
sns.pairplot(df_total)
plt.show()

##3-3.상관관계 분석
df_total.corr()

sns.heatmap(df_total.corr(), annot= True, cmap= 'RdBu')
plt.show()


##3-4.요인분석
###3-4-1. 라이브러리 import
# !pip install factor-analyzer
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo

###3-4-2. Kaiser-Meyer-Olkin(KMO) 검정 : 요인 분석을 위한 데이터의 적합성 측정(0.6 미만 부적절)
df_dropped= df.drop(['RID', 'TIME_DEPARTUREDATE', 'TIME_ARRIVEDATE'], axis=1)
df_dropped
kmo_all,kmo_model=calculate_kmo(df_dropped)
kmo_model

###3-4-3.ScreenPlot을 활용한 요인수 결정: Elbow 기법
fa = FactorAnalyzer()
fa.set_params(rotation=None)
fa.fit(df_dropped)
#고유값(eigenvalue):각각의 요인으로 설명할 수 있는 변수들의 분산 총합
ev, v = fa.get_eigenvalues()
ev

plt.scatter(range(1,df_dropped.shape[1]+1),ev)
plt.plot(range(1,df_dropped.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

###3-4-4.요인부하량 확인 및 시각화(0.4 이상 유의미, 0.5 이상 중요)
fa = FactorAnalyzer()
fa.set_params(n_factors=3, rotation=None)
fa.fit(df_dropped)
pd.DataFrame(fa.loadings_)

plt.figure(figsize=(6,10))
sns.heatmap(fa.loadings_, cmap="Blues", annot=True, fmt='.2f')

###3-4-5.크론바흐 계수(신뢰도) 계산(0.8 이상 양호)
def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=0, ddof=1)
    tscores = itemscores.sum(axis=1)
    nitems = itemscores.shape[1]
    return (nitems / (nitems-1)) * (1 - (itemvars.sum() / tscores.var(ddof=1)))

print(CronbachAlpha(df[['ET','ETA']]))
print(CronbachAlpha(df[['ET','A_DISTANCE']]))

###3-4-6.요인점수를 활용한 변수 생성
fa.transform(df_dropped)
