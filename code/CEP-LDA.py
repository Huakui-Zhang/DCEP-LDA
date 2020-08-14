# -*- coding: utf-8 -*-
"""
对数据集进行分词、去停用词、词形还原等处理；
实现了论文中CEP-LDA；
训练LDA；
基于维基百科，计算得到主题的主题连贯性C_v
"""

import os
import spacy
from spacy.lang.en import English
import datetime
import pickle as pk
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel_modify_line_36_39 import CoherenceModel
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_model(filename='./model.pk'):
    fr = open(filename, 'rb')
    model = pk.load(fr)
    fr.close()
    return model

def save_model(model, filename='./model.pk'):
    fw = open(filename, 'wb')
    pk.dump(model, fw)
    fw.close()
    
def getConcepts(entity_name):
    global freebase_dict
    global topn_concepts
    
    punc = ",.:;\"/\\'?<>{}[]+-()&%$@!^#*"
    s = ''
    for i in range(len(entity_name)):
        if i < len(entity_name) - 1 and entity_name[i] == '_' and entity_name[i + 1] in punc:
            s = s + ''
        else:
            s = s + entity_name[i]
    entity_name = s
    
    if entity_name.count('(') > entity_name.count(')'):
        entity_name = entity_name + ')'
#    entity_name = entity_name[len('ne_'):]
    entity_name = ' '.join(entity_name.split('_')[1:])
    if (entity_name in freebase_dict):
        concepts = freebase_dict[entity_name]
        if len(concepts) > topn_concepts:
            return concepts[0:topn_concepts]
        else:
            return concepts
    else:
        return []


def hasRelation(entity1, entity2):
    if entity1 == entity2:
        return True
    
    if len(set(getConcepts(entity1)) & set(getConcepts(entity2))) != 0:
        return True
    
    return False

def relationRatio(bow_news, ne_token_ids, id2token):
    entity_bow = [t for t in bow_news if t[0] in ne_token_ids]
    if len(entity_bow) == 0:
        return dict()

    entity_ids = [t[0] for t in entity_bow]
#    num_entity_type = len(entity_ids)
    num_entity = sum([t[1] for t in entity_bow])
    
    ratio = dict()
    for entity_id in entity_ids:
        entity_name = id2token[entity_id]
#        ratio[entity_id] = sum([t[1] for t in entity_bow if t[0] in entity_relation[entity_id]]) / num_entity
        ratio[entity_id] = sum([t[1] if hasRelation(entity_name, id2token[t[0]]) else 0 \
                               for t in entity_bow]) / num_entity
    
    return ratio

def relationNum(bow_news, ne_token_ids, id2token):
    entity_bow = [t for t in bow_news if t[0] in ne_token_ids]
    if len(entity_bow) == 0:
        return dict()

    entity_ids = [t[0] for t in entity_bow]
#    num_entity_type = len(entity_ids)
    num_entity = sum([t[1] for t in entity_bow])
    
    relation_num = dict()
    for entity_id in entity_ids:
        entity_name = id2token[entity_id]
#        ratio[entity_id] = sum([t[1] for t in entity_bow if t[0] in entity_relation[entity_id]]) / num_entity
        relation_num[entity_id] = sum([1 if hasRelation(entity_name, id2token[t[0]]) else 0 \
                               for t in entity_bow])
    
    return relation_num

starttime = datetime.datetime.now()

min_doc_tf = 3
min_word_char_num = 3

freebase_dict = load_model('D:/Study/SpyderProject/bishe/freebase/cached_info_50.dict')

#dataset = 'bbc' # 20news bbc reuters
# dataset_list = ['bbc', '20news', 'reuters']
dataset_list = ['reuters']
topn_concepts = 20

for dataset in dataset_list:
    data_dir = './%s_data'%dataset
    print('processing', dataset)
    with open(os.path.join(data_dir, 'ne.txt'), 'r', encoding='utf-8') as f:
        news_with_NE = f.readlines()
    news_with_NE = [news.strip() for news in news_with_NE if news.strip()]

    nlp = spacy.load('en', disable=['parser', 'ner'])
    tokenizer = English().Defaults.create_tokenizer(nlp)
    processed_news_list = []

    for news in news_with_NE:
        doc = tokenizer(news)
        # remove stop words, punctuations and words or NEs with too few characters and lemmatize and lower the doc
        processed_news = [str(token).lower() if str(token).startswith('ne_') else token.lemma_.lower() for token in doc \
                          if not token.is_punct and not nlp.vocab[str(token)].is_stop \
                                   and ((not str(token).startswith('ne_') and len(str(token)) >= min_word_char_num) or \
                                        (str(token).startswith('ne_') and len(str(token)) >= min_word_char_num + 3))]
        processed_news_list.append(processed_news)

    #processed_news_list = [news.split() for news in news_with_NE]

    dictionary = Dictionary(processed_news_list)
    # remove words with too few document frequency
    dictionary.filter_extremes(no_below=min_doc_tf)
    bow_news = [dictionary.doc2bow(doc) for doc in processed_news_list]
    bow_news = [news for news in bow_news if len(news)>0]

    # find the ids of the ne
    dict_token2id = dictionary.token2id
    dict_id2token = dict(dictionary.items())
    tokens = list(dict_token2id.keys())
    ne_tokens = [token for token in tokens if token.startswith('ne_')]
    ne_token_ids = [dict_token2id[token] for token in ne_tokens]
    ne_token_ids = set(ne_token_ids)

    # ne term weighting
    # add max token frequency tuple in documents
    bow_news = [news + [(-1, max([t[1] for t in news]))] for news in bow_news]

    bow_news = [news + [relationNum(news, ne_token_ids, dict_id2token)] for news in bow_news]

    bow_news = [[(t[0], t[1]+news[-2][1] * news[-1][t[0]]) if t[0] in ne_token_ids else (t[0], t[1]) \
                  for t in news[:-2]] for news in bow_news]

#     dictionary.save(os.path.join(data_dir, 'ne8_%s_%s_%s_weighting.dict'%(topn_concepts, gamma,lambd)))
    save_model(bow_news, os.path.join(data_dir, 'ne8_nedf_%s_weighting.bow'%(topn_concepts)))

    endtime = datetime.datetime.now()
    print('Totol running for ', (endtime - starttime).seconds, ' seconds.')



"""
加载需要的Dictionary和bag-of-words文件，调用Gensim中的LDA库训练LDA，每种主题数设置做5词实验
"""
import os
import pickle as pk
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
import datetime
from pprint import pprint
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_model(filename='./model.pk'):
    fr = open(filename, 'rb')
    model = pk.load(fr)
    fr.close()
    return model

# dataset = 'reuters' # 20news bbc reuters
# dataset = ['bbc', 'reuters', '20news']
dataset = ['reuters', '20news']
# dataset = [ '20news']

# Set training parameters.
# num_topics = 100
# num_topics_list = [40,60,80]
num_topics_list = [20]
passes = 100
iterations = 50
eval_every = None
workers = 6
# random_state  = 42
topn_concepts = 20
random_state_list = [7,14,21,28,42]
# random_state_list = [42]

for data in dataset:
    for n_topics in num_topics_list:
        duration_list = []
        for random_state in random_state_list:
            starttime = datetime.datetime.now()
            print('dataset:', data, 'num_topics:', n_topics, 'random_state:', random_state)
            data_dir = './%s_data'%data
            dictionary = Dictionary.load(os.path.join(data_dir, 'ne_nedf_weighting.dict'))
            bow_news = load_model(os.path.join(data_dir, 'ne8_nedf_%s_weighting.bow')%(topn_concepts))
            dict_id2token = dict(dictionary.items())

            lda = LdaMulticore(bow_news, id2word=dict_id2token, num_topics=n_topics, passes=passes, iterations=iterations,\
                               eval_every=eval_every, workers=workers, random_state=random_state)

            name = 'ne8_nedf_%s_topic%s_passes%s_iteration%s_random%s' % (topn_concepts, n_topics, passes, iterations, random_state)
            result_dir = os.path.join(data_dir, name)
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            lda.save(os.path.join(result_dir, 'lda_model'))

            topics = lda.show_topics(num_topics=n_topics, num_words=20, log=False, formatted=False)
            with open(os.path.join(result_dir, 'topics.txt'), 'w', encoding='utf-8') as f:
                for topic in topics:
                    f.write('topic ' + str(topic[0]) + ':\n')
                    for t in topic[1]:
                        f.write(t[0] + ': ' + str(t[1]) + '\n')
                    f.write('\n')

            endtime = datetime.datetime.now()
            duration = (endtime - starttime).seconds
            duration_list.append(duration)
            print('Totol running for ', (endtime - starttime).seconds, ' seconds.')
        print(sum(duration_list)/len(duration_list))


'''
得到维基百科经过相同分词、去停用词、词形还原预处理操作后所有token的set
wiki_dir文件夹中存放的是预处理后的维基百科文件
基于维基百科，计算得到主题的主题连贯性C_v
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

wiki_dir = 'D:/Study/SpyderProject/bishe/enwiki-20190301/ne/lemma1'
token_set = set()
token_set_path = './token_set.pk'
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
    save_model(token_set, './token_set.pk')
print('load token set')

# dataset_list = ['20news', 'bbc', 'reuters']
dataset_list = ['20news','reuters']
topic_num_list = [40,60,80]
# topic_num_list = [200]
passes = 100
processes = 2
topn = 10
topn_concepts = 20
random_state_list = [7,14,21,28,42]

for dataset in dataset_list:
    starttime = datetime.datetime.now()
    dict_path = 'D:/Study/JupyterProject/bishe/%s_data/ne_nedf_weighting.dict'%dataset
    dictionary = Dictionary.load(dict_path)
    all_topics = []
    
    for topic_num in topic_num_list:
        for random_state in random_state_list:
            print('dataset:', dataset, 'topic_num:', topic_num, 'random_state:', random_state)
            
            result_dir = 'D:/Study/JupyterProject/bishe/%s_data/ne8_nedf_%s_topic%s_passes%s_iteration50_random%s'%(dataset, topn_concepts, topic_num, passes, random_state)

            lda_model_path = os.path.join(result_dir, 'lda_model')
            lda_model = LdaMulticore.load(lda_model_path)
            topics = lda_model.show_topics(num_topics=topic_num, num_words=topn*10, log=False, formatted=False)
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
            result_dir = 'D:/Study/JupyterProject/bishe/%s_data/ne8_nedf_%s_topic%s_passes%s_iteration50_random%s'%(dataset, topn_concepts, topic_num, passes, random_state)
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
