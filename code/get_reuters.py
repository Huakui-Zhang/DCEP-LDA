#import nltk
#nltk.download('reuters')
from nltk.corpus import reuters
import os
import re
import pandas as pd

def process_news(news):
    news = re.sub("&lt;", ' ', news)
    news = re.sub(">", ' ', news)
    news = news.split('\n')
    # add period for news title
    news[0] = news[0] + '.'
    news = [' '.join(s.split()) for s in news if s.strip()]
    news = ' '.join(news)
    return news

reuters_dataset = []
for id in reuters.fileids():
    news = reuters.raw(id)
    news = ''.join(news)
    reuters_dataset.append(news)
    
reuters_dataset = list(map(process_news, reuters_dataset))

data_dir = './reuters_data'
# write each piece of news as a document for NER
seperate_data_dir = os.path.join(data_dir, 'reuters_raw/deploy')
if not os.path.exists(seperate_data_dir):
    os.makedirs(seperate_data_dir)

for i in range(len(news)):
   with open(os.path.join(seperate_data_dir, '%s.txt'%i), 'w', encoding='utf-8') as f:
       f.write(news[i])

with open(os.path.join(data_dir, 'raw.txt'), 'w', encoding='utf-8') as f:
   f.write('\n'.join(news))