# -*- coding: utf-8 -*-
"""
对数据集进行分词、去停用词、词形还原等处理；
训练LDA；
基于维基百科，计算得到主题的主题连贯性C_v
"""

import os
import spacy
from spacy.lang.en import English
import datetime
import pickle as pk
from gensim.corpora import Dictionary
from collections import defaultdict

def load_model(filename='./model.pk'):
    fr = open(filename, 'rb')
    model = pk.load(fr)
    fr.close()
    return model

def save_model(model, filename='./model.pk'):
    fw = open(filename, 'wb')
    pk.dump(model, fw)
    fw.close()

starttime = datetime.datetime.now()
# minimum document frequency of a term
min_doc_tf = 3
# minimum number of characters of a word/term
min_word_char_num = 3

dataset_list = ['20news', 'reuters']

for dataset in dataset_list:
    data_dir = './%s_data'%dataset
    print('processing', dataset)
    with open(os.path.join(data_dir, 'raw.txt'), 'r', encoding='utf-8') as f:
        news_with_NE = f.readlines()
    news_with_NE = [news.strip() for news in news_with_NE if news.strip()]

    nlp = spacy.load('en', disable=['parser', 'ner'])
    tokenizer = English().Defaults.create_tokenizer(nlp)
    processed_news_list = []

    for news in news_with_NE:
        doc = tokenizer(news)
        # remove stop words, punctuations and words or NEs with too few characters and lemmatize and lower the doc
        processed_news = [token.lemma_.lower() for token in doc \
                          if not token.is_punct and not nlp.vocab[str(token)].is_stop \
                                    and len(str(token)) >= min_word_char_num]
        processed_news_list.append(processed_news)

    #processed_news_list = [news.split() for news in news_with_NE]

    dictionary = Dictionary(processed_news_list)

    # remove words with too few document frequency
    dictionary.filter_extremes(no_below=min_doc_tf)
    bow_news = [dictionary.doc2bow(doc) for doc in processed_news_list]
    bow_news = [news for news in bow_news if len(news)>0]
    print(len(bow_news))

    dictionary.save(os.path.join(data_dir, 'lda.dict'))
    save_model(bow_news, os.path.join(data_dir, 'lda.bow'))

    endtime = datetime.datetime.now()
    print('Totol running for ', (endtime - starttime).seconds, ' seconds.')


"""
加载需要的Dictionary和bag-of-words文件，调用Gensim中的LDA库训练LDA
"""

import os
import pickle as pk
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
import datetime
from pprint import pprint

def load_model(filename='./model.pk'):
    fr = open(filename, 'rb')
    model = pk.load(fr)
    fr.close()
    return model

dataset = ['reuters', '20news']

# Set training parameters.
num_topics_list = [60, 80]
passes_list = [100]
passes = 100
iterations = 50
eval_every = None
workers = 2
random_state_list = [7,14,21,28,42]

for data in dataset:
    for n_topics in num_topics_list:
        for random_state in random_state_list:
            starttime = datetime.datetime.now()
            print('dataset:', data, 'num_topics:', n_topics, 'random_state:', random_state)
            data_dir = './%s_data'%data
            dictionary = Dictionary.load(os.path.join(data_dir, 'lda.dict'))
            bow_news = load_model(os.path.join(data_dir, 'lda.bow'))
            dict_id2token = dict(dictionary.items())

            lda = LdaMulticore(bow_news, id2word=dict_id2token, num_topics=n_topics, passes=passes, iterations=iterations,\
                               eval_every=eval_every, workers=workers, random_state=random_state)

            name = 'lda_topic%s_passes%s_iteration%s_random%s' % (n_topics, passes, iterations, random_state)
            result_dir = os.path.join(data_dir, name)
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            lda.save(os.path.join(result_dir, 'lda_model'))

            topics = lda.show_topics(num_topics=n_topics, num_words=10, log=False, formatted=False)
            # 输出主题
            with open(os.path.join(result_dir, 'topics.txt'), 'w', encoding='utf-8') as f:
                for topic in topics:
                    f.write('topic ' + str(topic[0]) + ':\n')
                    for t in topic[1]:
                        f.write(t[0] + ': ' + str(t[1]) + '\n')
                    f.write('\n')

            endtime = datetime.datetime.now()
            print('Totol running for ', (endtime - starttime).seconds, ' seconds.')

'''
得到维基百科经过相同分词、去停用词、词形还原预处理操作后所有token的set
wiki_dir文件夹中存放的是讲过同样预处理（去停用词等，及NER）后的维基百科文件，用户可自行实现
'''
import os
import datetime
import pickle as pk
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel_modify_line_36_39 import CoherenceModel


def load_model(filename='./model.pk'):
    fr = open(filename, 'rb')
    model = pk.load(fr)
    fr.close()
    return model

def save_model(model, filename='./model.pk'):
    fw = open(filename, 'wb')
    pk.dump(model, fw)
    fw.close()

wiki_dir = 'E:\\wiki_data\\lemma'
token_set = set()
token_set_path = './raw_token_set.pk'
if os.path.exists(token_set_path):
    token_set = load_model(token_set_path)
else:
    for file_name in os.listdir(wiki_dir):
        print('file:', file_name)
        with open(os.path.join(wiki_dir, file_name), 'rb') as f:
            lines = pk.load(f)
        for line in lines:
            for token in line:
                token_set.add(token)
        lines = []
    save_model(token_set, token_set_path)
print('load token set')      


'''
基于维基百科，计算得到主题的主题连贯性C_v
'''
dataset_list = ['20news','reuters']
topic_num_list = [40, 60, 80]
passes = 100
processes = 2
random_state_list = [7,14,21,28,42]
topn = 10

for dataset in dataset_list:
    starttime = datetime.datetime.now()
    dict_path = 'D:/Study/JupyterProject/bishe/%s_data/lda.dict'%dataset
    dictionary = Dictionary.load(dict_path)
    all_topics = []
    
    for topic_num in topic_num_list:
        for random_state in random_state_list:          
            print('dataset:', dataset, 'topic_num:', topic_num, 'random_state:', random_state)

            result_dir = 'D:/Study/JupyterProject/bishe/%s_data/lda_topic%s_passes%s_iteration50_random%s'%(dataset,  topic_num, passes, random_state)
            lda_model_path = os.path.join(result_dir, 'lda_model')
            lda_model = LdaMulticore.load(lda_model_path)
            topics = lda_model.show_topics(num_topics=topic_num, num_words=topn*2, log=False, formatted=False)
            topics_clean = [[word[0] for word in topic[1] if word[0] in token_set] for topic in topics]
            topics_clean = [topic if len(topic) < topn else topic[0:topn] for topic in topics_clean]
            all_topics.extend(topics_clean)

    cm = CoherenceModel(topics=all_topics, dictionary=dictionary, texts=wiki_dir, coherence='c_v', window_size=110, topn=topn, processes=processes)
    all_coherence_per_topic = cm.get_coherence_per_topic()  # get coherence value

    print('write')
    for topic_num in topic_num_list:
        all_coherence = []
        all_coherence_str = []
        for random_state in random_state_list:
            result_dir = 'D:/Study/JupyterProject/bishe/%s_data/lda_topic%s_passes%s_iteration50_random%s'%(dataset,  topic_num, passes, random_state)
            coherence_per_topic = all_coherence_per_topic[0:topic_num]
            mean_coherence_per_topic = np.mean(coherence_per_topic)
            all_coherence.append(mean_coherence_per_topic)
            all_coherence_str.append(str(mean_coherence_per_topic))
            all_coherence_per_topic = all_coherence_per_topic[topic_num:]
            with open(os.path.join(result_dir, 'coherence_topn%s_%s.txt'%(topn, mean_coherence_per_topic)), 'w', encoding='utf-8') as f:
                f.write(' '.join([str(i) for i in coherence_per_topic]))
                f.write('\n')
                f.write(str(mean_coherence_per_topic))
            
            topics = all_topics[0:topic_num]
            all_topics = all_topics[topic_num:]

            coherence_topic = list(zip(coherence_per_topic, topics))
            coherence_topic = sorted(coherence_topic, key = lambda coherence: coherence[0], reverse = True)
            coherence_topic = [str(coherence[0]) + ': ' + ' '.join(coherence[1]) for coherence in coherence_topic]
            with open(os.path.join(result_dir, 'topic_coherence.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(coherence_topic))
                
        print(dataset, topic_num, 'coherence:', ' '.join(all_coherence_str), np.mean(all_coherence))

    endtime = datetime.datetime.now()
    print('Totol running for ', (endtime - starttime).seconds, ' seconds.')