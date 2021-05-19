"""
数据预处理
"""

import os
import pandas as pd
import xml.etree.cElementTree as ET


laptop_path = 'datasets/Laptop_Train_v2.xml'
res_path = 'datasets/Restaurants_Train_v2.xml'


# laptop
tree = ET.parse(laptop_path)
root = tree.getroot()

data = []
for sentence in root.findall('.//aspectTerms/..'):
    text = sentence.find('text').text
    aspectTerms = sentence.find('aspectTerms')
    for aspectTerm in aspectTerms.findall('aspectTerm'):
        term = aspectTerm.get('term')
        polarity = aspectTerm.get('polarity')
        data.append((text, term, polarity))

df1 = pd.DataFrame(data, columns=['text', 'term', 'polarity'])
df1 = df1[df1['polarity'].isin(['positive', 'negative', 'neutral'])]
df1['polarity'] = df1['polarity'].map(
    {'positive': 1, 'neutral': 0, 'negative': -1})
if not os.path.exists('output'):
    os.makedirs('output')
df1.to_csv('output/laptop_data.csv', encoding='utf-8')
print('*' * 50)
print(df1.head())

# split train and test
length = len(df1)
threshold = length // 5
train = df1[:-threshold]
test = df1[-threshold:]
train.to_csv('output/laptop_data_train.csv', encoding='utf-8')
test.to_csv('output/laptop_data_test.csv', encoding='utf-8')


# restaurant
tree = ET.parse(res_path)
root = tree.getroot()

data = []
for sentence in root.findall('.//aspectTerms/..'):
    text = sentence.find('text').text
    aspectTerms = sentence.find('aspectTerms')
    for aspectTerm in aspectTerms.findall('aspectTerm'):
        term = aspectTerm.get('term')
        polarity = aspectTerm.get('polarity')
        data.append((text, term, polarity))

df2 = pd.DataFrame(data, columns=['text', 'term', 'polarity'])
df2 = df2[df2['polarity'].isin(['positive', 'negative', 'neutral'])]
df2['polarity'] = df2['polarity'].map(
    {'positive': 1, 'neutral': 0, 'negative': -1})
if not os.path.exists('output'):
    os.makedirs('output')
df2.to_csv('output/res_data.csv', encoding='utf-8')
print('*' * 50)
print(df2.head())

# split train and test
length = len(df2)
threshold = length // 5
train = df2[:-threshold]
test = df2[-threshold:]
train.to_csv('output/res_data_train.csv', encoding='utf-8')
test.to_csv('output/res_data_test.csv', encoding='utf-8')






