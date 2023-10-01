#1.데이터 불러오기
import pandas as pd
import numpy as np

df_feature = pd.read_csv("onenavi_train_feature.csv",sep="|")
df_target = pd.read_csv("onenavi_train_target.csv",sep="|")

#2.Train 데이터/ Test 데이터 분리
# !pip install sklearn
from sklearn.model_selection import train_test_split 

train_x, test_x, train_y, test_y= train_test_split(df_feature, df_target, test_size= 0.2, random_state=42)
train_x

#3.모델링
##3-1.Linear Regression
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

model= lr()
model.fit(train_x, train_y)
print("회귀계수: ", model.coef_, ", 절편: ", model.intercept_)

pred_y= model.predict(test_x)
print("RMSE: ",mean_squared_error(test_y, pred_y)**0.5)
print("R-Squared Score: ", r2_score(test_y, pred_y))

##3-2.Random Tree Forest Regression
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

train_y= np.ravel(train_y, order= 'C')

model= rfr(n_estimators= 100, max_depth=5, min_samples_split=30, min_samples_leaf= 15)
model.fit(train_x, train_y)

pred_y= model.predict(test_x)

print("RMSE: ",mean_squared_error(test_y, pred_y)**0.5)
print("R-squared score: ", r2_score(test_y, pred_y))

#feature의 중요도 확인
import matplotlib.pyplot as plt
# !pip install seaborn
import seaborn as sns

rf_importances_values= model.feature_importances_
rf_importances= pd.Series(rf_importances_values, index= train_x.columns)
rf_top10= rf_importances.sort_values(ascending= False) [:10]

plt.rcParams["font.family"]= 'NanumGothicCoding'
plt.figure(figsize=(8,6))
plt.title('Top 10 Feature Importances')
sns.barplot(x= rf_top10, y= rf_top10.index, palette= "RdBu")
plt.show()

##3-3.Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor as grb
# from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

train_y= np.ravel(train_y, order= 'C')

model= grb(n_estimators= 100, learning_rate= 0.1, max_depth= 5, min_samples_split= 30, min_samples_leaf= 15)
model.fit(train_x, train_y)

pred_y= model.predict(test_x)

print("RMSE: ",mean_squared_error(test_y, pred_y)**0.5)
print("R-squared score: ", r2_score(test_y, pred_y))

import seaborn as sns
import matplotlib.pyplot as plt

grb_importances_values= model.feature_importances_
grb_importances= pd.Series(grb_importances_values, index= train_x.columns)
grb_top10= grb_importances.sort_values(ascending= False) [:10]

plt.rcParams["font.family"]= 'NanumGothicCoding'
plt.figure(figsize= (8,6))
plt.title('Top 10 Feature Importances')
sns.barplot(x= grb_top10, y= grb_top10.index, palette= "RdBu")
plt.show()


##3-4.XGBoosting
from xgboost import XGBRegressor as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

train_y= np.ravel(train_y, order= 'C')

model= xgb(n_estimators= 100, gamma= 1, eta= 0.1, max_depth= 5, reg_lambda= 5, reg_alpha=5)
model.fit(train_x, train_y)

pred_y= model.predict(test_x)

print("RMSE: ",mean_squared_error(test_y, pred_y)**0.5)
print("R-squared score: ", r2_score(test_y, pred_y))

import seaborn as sns
import matplotlib.pyplot as plt

xgb_importances_values= model.feature_importances_
xgb_importances= pd.Series(xgb_importances_values, index= train_x.columns)
xgb_top10= xgb_importances.sort_values(ascending= False)[:10]

plt.rcParams['font.family']= 'NanumGothicCoding'
plt.figure(figsize= (8,6))
plt.title('Top 10 Feature Importances')
sns.barplot(x= xgb_top10, y= xgb_top10.index, palette= "RdBu")
plt.show()

#4. 모델 저장 및 비교
# !pip install sklearn
from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import GradientBoostingRegressor as grb
from xgboost import XGBRegressor as xgb 

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

import  pickle
import joblib
import time

model_list= [
    lr(),
    rfr(),
    grb(),
    xgb(),
]

train_y= np.ravel(train_y, order= 'C')

model_results= []

for i in range(len(model_list)):
    start_time= time.process_time()
    model= model_list[i]
    model.fit(train_x, train_y)
    end_time= time.process_time()
    joblib.dump(model, '{}_model.pkl'.format(i))
    print(f"*{model} 결과")
    print('---- {0:.5f}sec, training complete ----'.format(end_time-start_time))
    pred_y= model.predict(test_x)
    model_results.append(model)
    print("RMSE on test set: {0:.5f}".format(mean_squared_error(test_y, pred_y)**0.5))
    print("R-squared score on test set : {0:.5f}".format(r2_score(test_y, pred_y)))
    print('-------------------------------------------------------------------------')




