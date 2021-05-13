import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.72 Safari/537.36'
}

url = 'https://coding.imooc.com/class/evaluation/414.html?page='

urls = []
for i in range(10):
    urls.append(url+str(i))

texts = []
labels = []
terms = []

for url in tqdm(urls):
    soup = None
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'html.parser')
    comments = soup.find('ul', attrs={'class':'cmt-list'}).find_all('li', attrs={'class':'cmt-post elist-wrap'})
    for comment in comments:
        text = comment.find('p', attrs={'class':'cmt-txt'}).get_text()
        polarity = comment.find('div', attrs={'class':'stars'}).span.get_text()
        texts.append(text)
        labels.append(polarity)
        terms.append('NAN')

df = pd.DataFrame()
df['text'] = texts
df['term'] = terms
df['polarity'] = labels
df.to_csv('output/mooc_data.csv', encoding='utf-8')



# split train and test
df2 = pd.read_csv('output/mooc_data.csv', encoding='utf-8')
length = len(df2)
threshold = length // 5
train = df2[:-threshold]
test = df2[-threshold:]
train.to_csv('output/mooc_data_train.csv', encoding='utf-8')
test.to_csv('output/mooc_data_test.csv', encoding='utf-8')
