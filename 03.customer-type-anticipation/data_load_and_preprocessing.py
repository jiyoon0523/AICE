#0. 라이브러리 임포트
import warnings
warnings.filterwarnings('ignore')
import subprocess
import sys
import pandas as pd
import numpy as np
import pickle
import hashlib

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("folium")
install("seaborn")
install("xgboost")

#1. 데이터 수집
##1-1. 데이터 암호화
ansan_data= pd.read_csv('ansan_data.csv')
ansan_data.head()

def encrypt(target):
    hashSHA= hashlib.sha256() 
    hashSHA.update(str(target).encode('utf-8'))
    return hashSHA.hexdigest().upper() 

encrypt('홍길동')

ansan_data['IS']= ansan_data['IS'].apply(encrypt)
ansan_data['JOIN_SEQ']= ansan_data['JOIN_SEQ'].apply(encrypt)
ansan_data.head()

##1-2. 주소에 대한 위경도 좌표 수집
with open('json_data.pickle', 'rb') as f:
    json_data = pickle.load(f)

json_data[:10]
json_data[0].json()
json_data[0].json()['documents']
json_data[0].json()['documents'][0]

json_data[0].json()['documents'][0]['address_name']
json_data[0].json()['documents'][0]['x'], json_data[0].json()['documents'][0]['y'] 

address_data= pd.DataFrame([])

for i in np.arange(len(json_data)):
    if json_data[i].json()['documents']!=[]:
        address_data= address_data.append([(json_data[i].json()['documents'][0]['address_name'],
                             json_data[i].json()['documents'][0]['road_address']['x'],
                             json_data[i].json()['documents'][0]['road_address']['y'])],
                             ignore_index= False) 
        
address_data

ansan_data.iloc[:, -3:]
ansan_data.head(3)

