# !pip install bs4
# !pip install openpyxl

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse

#1.데이터 로드
filename ='Feature Website.xlsx'
df = pd.read_excel(filename, engine='openpyxl')
df

#2.html에서 <script>...</script>길이 계산
def html_script_characters(soup):
    html_str= str(soup.script)
    return float(len(html_str.replace(' ','')))

script_len= []

for index, row in df.iterrows():
    soup= BeautifulSoup(row.html_code, 'html.parser')
    script_len.append(html_script_characters(soup))

df['html_script_characters'] = script_len
df.describe()

#3.html에서 공백 수 계산
def html_num_whitespace(soup):
    try:
        NullCount = soup.body.text.count(' ')
        return float(NullCount)
    except:
        return 0.0
    
num_whitespace = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    num_whitespace.append(html_num_whitespace(soup))

df['html_num_whitespace'] = num_whitespace
df.describe()

#4.html에서 body길이 계산
def html_num_characters(soup):
    try:
        bodyLen = len(soup.body.text)
        return float(bodyLen)
    except:
        return 0.0

html_body = []

for index, row in df.iterrows():
    soup =  BeautifulSoup(row.html_code, 'html.parser')
    html_body.append(html_num_characters(soup))

df['html_body_len'] = html_body
df.describe()

#5.script에서 src, href 속성을 가진 태그 수
def html_link_in_script(soup):
    numOfLinks = len(soup.findAll('script', {"src": True}))
    numOfLinks += len(soup.findAll('script',{"href": True}))
    return float(numOfLinks)

html_script_link_num = []

for index, row in df.iterrows():
    soup =  BeautifulSoup(row.html_code, 'html.parser')
    html_script_link_num.append(html_link_in_script(soup))

df['html_script_link_num'] = html_script_link_num
df.describe()
