#pip install pandas
import pandas as pd
import numpy as np
#pip install ipython
from IPython.display import Image

#1.1 DataFrame 만들어 보기
##Dictionary
a1= pd.DataFrame({"a":[1,2,3], "b": [4,5,6], "c": [7,8,9]})
print(a1)

##List
a2= pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], ["a", "b", "c"])
print(a2)

##Load Data
cust= pd.read_csv('./sc_cust_info_txn_v1.5.csv', encoding= "cp949")
print(cust)
cust.head(n=3) #0, 1, 2 열 
cust.tail(n=10) #N, N-1, ..., N-9 열

#shape
cust.shape
cust.columns
cust.info
cust.describe
cust.dtypes



