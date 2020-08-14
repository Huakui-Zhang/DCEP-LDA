# -*- coding: utf-8 -*-
"""
简单预处理20Newsgroups数据集，把所有新闻输入为一个文件，另外将每一条新闻写入一个单独的文件中，用于后续NER处理。
"""

import os
from sklearn.datasets import fetch_20newsgroups

data_dir = './20news_data'
# directory for further NER use
seperate_data_dir = os.path.join(data_dir, '20news_raw/deploy')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(seperate_data_dir):
    os.makedirs(seperate_data_dir)

dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

data = dataset.data
target = dataset.target
data = zip(data, target)

# remove documents that are full of arbitrary characters
data = [news for news in data if news.count('@') < 20 and news.count('%') < 20 and news.count('&') < 20]

# write each piece of news as a document for NER
for i in range(len(data)):
   with open(os.path.join(seperate_data_dir, '%s.txt'%i), 'w', encoding='utf-8') as f:
       f.write(data[i])

# write all the news in a document
with open(os.path.join(data_dir, 'raw.txt'), 'w', encoding='utf-8') as f:
   f.write('\n'.join(data))
