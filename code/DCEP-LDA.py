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
    global concept_topn
    
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
        if len(concepts) > concept_topn:
            return concepts[0:concept_topn]
        else:
            return concepts
    else:
        return []


def hasRelation(entity1, entity2):
#    global num_relations_in_new_per_entity
    if entity1 == entity2:
#        num_relations_in_new_per_entity += 1
        return True
    
    if len(set(getConcepts(entity1)) & set(getConcepts(entity2))) != 0:
#        num_relations_in_new_per_entity += 1
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

def entity_cluster(bow_news, ne_token_ids, id2token, is_dupl, min_cluster_size):
    entity_bow = [t for t in bow_news if t[0] in ne_token_ids]
    if len(entity_bow) < 2:
        return []
    
    entity_ids = [t[0] for t in entity_bow]
#    entity_names = [id2token[i] for i in entity_ids]
    new_d = defaultdict(list)
    for i in entity_ids:
        for c in getConcepts(id2token[i]):
            new_d[c].append(i)
        
    new_d = dict(new_d)
    cluster_list = []
    for k, v in new_d.items():
        if len(v) > min_cluster_size:
            cluster_list.append(v)
            
    if not is_dupl:
        cluster_set = set(tuple(sorted(x)) for x in cluster_list)
        cluster_list = [list(x) for x in cluster_set]
    
    return cluster_list

def relation_cluster(bow_news, ne_token_ids, id2token, is_dupl, min_cluster_size):
    entity_bow = [t for t in bow_news if t[0] in ne_token_ids]
    if len(entity_bow) < 2:
        return []

    entity_ids = [t[0] for t in entity_bow]
    
    d = dict()
    for entity_id in entity_ids:
        entity_name = id2token[entity_id]
        d[entity_id] = [t[0] for t in entity_bow if hasRelation(entity_name, id2token[t[0]])]
    
    cluster_list = []
    for k, v in d.items():
        if len(v) > min_cluster_size:
            cluster_list.append(v)
    
    if not is_dupl:
        cluster_set = set(tuple(sorted(x)) for x in cluster_list)
        cluster_list = [list(x) for x in cluster_set]
    
    return cluster_list

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
dataset_list = ['20news', 'reuters']
# dataset_list = ['reuters']
# min_cluster_size_list = [2,3,4,5]
min_cluster_size_list = [1]
concept_topn = 20
is_dupl = True


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
    print('bow_news size:', len(bow_news))
    
    # find the ids of the ne
    dict_token2id = dictionary.token2id
    dict_id2token = dict(dictionary.items())
    tokens = list(dict_token2id.keys())
    ne_tokens = [token for token in tokens if token.startswith('ne_')]
    ne_token_ids = [dict_token2id[token] for token in ne_tokens]
    ne_token_ids = set(ne_token_ids)
    
    for min_cluster_size in min_cluster_size_list:
        print('min_cluster_size:', min_cluster_size)
        # new entity term weighting
        new_bow_news = []
        for news in bow_news:
            max_frequency = max([t[1] for t in news])
            en_ids_in_news = [t[0] for t in news if t[0] in ne_token_ids]
            if len(en_ids_in_news) == 0:
                new_bow_news.append(news)
                continue
            elif len(en_ids_in_news) == 1:
                new_news = [(t[0], t[1] + max_frequency) if t[0] in en_ids_in_news else t for t in news]
                new_bow_news.append(new_news)
                continue

            new_news_list = []
            cluster = entity_cluster(news, ne_token_ids, dict_id2token, is_dupl, min_cluster_size)
            cluster_size = len(cluster)
            relation_n = relationNum(news, ne_token_ids, dict_id2token)
            if cluster_size < 2:
                new_news = [(t[0], t[1] + max_frequency*relation_n[t[0]]) if t[0] in ne_token_ids else (t[0],t[1]) for t in news]
                new_bow_news.append(new_news)
                continue

            for c in cluster:
                new_news = [(t[0], t[1] + max_frequency*len(c)) if t[0] in c else ((t[0],(t[1] + max_frequency)) if t[0] in ne_token_ids else (t[0],t[1]))  for t in news]
                new_news_list.append(new_news)
            new_bow_news.extend(new_news_list)


        print('new_bow_news size:', len(new_bow_news))
        save_model(new_bow_news, os.path.join(data_dir, 'ne10_nedf_%s_%s_maxMin_addNew_cSizelt%s_weighting.bow')%(concept_topn, 'dupl' if is_dupl else 'noDupl', min_cluster_size))

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

dataset_list = ['20news', 'reuters']
# dataset_list = ['20news']
# min_cluster_size_list = [2,3,4,5]
min_cluster_size_list = [1]
# Set training parameters.
# num_topics = 100
# num_topics_list = [20]
num_topics_list = [20,40,60,80,100]
passes = 200
iterations = 50
eval_every = None
workers = 6
# random_state  = 42
random_state_list = [7,14,21,28,42]
# random_state_list = [42]


top_n_concepts= 20
is_dupl = True

for data in dataset_list:
    for n_topics in num_topics_list:
        for min_cluster_size in min_cluster_size_list:
            duration_list = []
            for random_state in random_state_list:                 
                starttime = datetime.datetime.now()
                print('dataset:', data, 'num_topics:', n_topics, 'min_cluster_size:', min_cluster_size, 'random_state:', random_state)
                data_dir = './%s_data'%data
                dictionary = Dictionary.load(os.path.join(data_dir, 'ne_nedf_weighting.dict'))
                bow_news = load_model(os.path.join(data_dir, 'ne10_nedf_%s_%s_maxMin_addNew_cSizelt%s_weighting.bow'%(top_n_concepts, 'dupl' if is_dupl else 'noDupl', min_cluster_size)))
                dict_id2token = dict(dictionary.items())

                lda = LdaMulticore(bow_news, id2word=dict_id2token, num_topics=n_topics, passes=passes, iterations=iterations,\
                                   eval_every=eval_every, workers=workers, random_state=random_state)
                #lda = LdaModel(bow_news, id2word=dict_id2token, num_topics=num_topics, passes=passes, iterations=iterations,\
                #                   eval_every=eval_every, random_state=random_state)

                #print(lda.show_topics(num_topics=num_topics, num_words=20))

                name = 'ne10_nedf_%s_%s_maxMin_addNew_cSizelt%s_topic%s_passes%s_iteration%s_random%s' % (top_n_concepts, 'dupl' if is_dupl else 'noDupl', min_cluster_size, n_topics, passes, iterations, random_state)
                result_dir = os.path.join(data_dir, name) 
                if not os.path.exists(result_dir):
                    os.mkdir(result_dir)

                lda.save(os.path.join(result_dir, 'lda_model'))

                topics = lda.show_topics(num_topics=n_topics, num_words=20, log=False, formatted=False)
                # 输出主题
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
# wiki_dir = 'D:/Study/SpyderProject/bishe/enwiki-20190301/ne/test'
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
                
dataset_list = ['20news','reuters']
dataset_list = ['20news']
# min_cluster_size_list = [2,3,4,5]
min_cluster_size_list = [1]
# topic_num_list = [40,60,80]
topic_num_list = [20,40,60,80,100]
passes = 200
processes = -1
random_state_list = [7,14,21,28,42]
# random_state_list = [42]

topn = 10
topn_concepts = 20
is_dupl = True

for dataset in dataset_list:
    starttime = datetime.datetime.now()
    dict_path = 'D:/Study/JupyterProject/bishe/%s_data/ne_nedf_weighting.dict'%dataset
    dictionary = Dictionary.load(dict_path)
    all_topics = []
    
    for topic_num in topic_num_list:
        for min_cluster_size in min_cluster_size_list:
            for random_state in random_state_list:          
                print('dataset:', dataset, 'topic_num:', topic_num, 'min_cluster_size:', min_cluster_size, 'random_state:', random_state)

                result_dir = 'D:/Study/JupyterProject/bishe/%s_data/ne10_nedf_%s_%s_maxMin_addNew_cSizelt%s_topic%s_passes%s_iteration50_random%s'%(dataset, topn_concepts, 'dupl' if is_dupl else 'noDupl', min_cluster_size, topic_num, passes, random_state)
                lda_model_path = os.path.join(result_dir, 'lda_model')
                lda_model = LdaMulticore.load(lda_model_path)
                topics = lda_model.show_topics(num_topics=topic_num, num_words=topn*2, log=False, formatted=False)
                topics_clean = [[word[0] for word in topic[1] if word[0] in token_set] for topic in topics]
                topics_clean = [topic if len(topic) < topn else topic[0:topn] for topic in topics_clean]
                all_topics.extend(topics_clean)

    cm = CoherenceModel(topics=all_topics, dictionary=dictionary, texts=wiki_dir, coherence='c_v', window_size=110, topn=topn, processes=processes)
    all_coherence_per_topic = cm.get_coherence_per_topic()  # get coherence value
    
#     with open(os.path.join('D:/Study/JupyterProject/bishe/%s_data'%dataset, 'coherence_topn%s.txt'%topn), 'w', encoding='utf-8') as f:
#             f.write(' '.join([str(i) for i in all_coherence_per_topic]))
    print('write')
    for topic_num in topic_num_list:
        for min_cluster_size in min_cluster_size_list:
            all_coherence = []
            all_coherence_str = []
            for random_state in random_state_list:
                result_dir = 'D:/Study/JupyterProject/bishe/%s_data/ne10_nedf_%s_%s_maxMin_addNew_cSizelt%s_topic%s_passes%s_iteration50_random%s'%(dataset, topn_concepts, 'dupl' if is_dupl else 'noDupl', min_cluster_size, topic_num, passes, random_state)             
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
                    
            print(dataset, topic_num, min_cluster_size,  'coherence:', ' '.join(all_coherence_str), np.mean(all_coherence))

    endtime = datetime.datetime.now()
    print('Totol running for ', (endtime - starttime).seconds, ' seconds.')
