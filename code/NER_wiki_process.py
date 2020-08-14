# -*- coding: utf-8 -*-
"""
处理NeuroNER对维基百科识别产生的BRAT格式的文件，对识别出的实体使用下划线连接，
并在实体前加上"ne"以作标注，如"ne_Donald_Trump"，输出文档文本文件
"""
import os
import datetime
import pandas as pd

file_list = list(range(0,2550))
for index in file_list:
    starttime = datetime.datetime.now()
    print('processing file', index)
    news_dir = './enwiki-20190301/seperate_preprocess_data'
    news_dir = os.path.join(news_dir, str(index))
    ner_dir = './NeuroNER-master/output/wiki/'
    output_dir = './enwiki-20190301/ne/seperate'
    
    with open(os.path.join(ner_dir, str(index) + '.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    NER_results = [i.split() for i in lines if i.strip()]
    
    NER_df = pd.DataFrame(NER_results, columns=['word', 'file', 'start', 'end', 'others', 'result'])
    NER_df = NER_df.apply(pd.to_numeric, errors='ignore')
    NER_df = NER_df.sort_values(['file', 'start'], 0, False)
    
    news = []
    with open(os.path.join(news_dir, '0.txt'), 'r', encoding='utf-8') as f:
        news = f.readlines()
    news = [n.strip() for n in news]
    
    beginning_flag = ['B-ORG', 'B-PER', 'B-LOC', 'B-MISC']
    inside_flag = ['I-ORG', 'I-PER', 'I-LOC', 'I-MISC']
    
    current_file = len(news) - 1
    print(len(news))
    
    punc = ",.:;\"/\\'?<>{}[]+-()&%$@!^#*"
    for i in range(0, len(NER_df)):
        if NER_df.iloc[i]['result'] in beginning_flag:
            file_num = NER_df.iloc[i]['file']
            if file_num//100 != current_file//100 and file_num%100 == 0:
                print(file_num)
                current_file = file_num
            start = NER_df.iloc[i]['start']
            end = NER_df.iloc[i]['end']
            for j in range(i - 1, -1, -1):
                if NER_df.iloc[j]['result'] in inside_flag:
                    end = NER_df.iloc[j]['end']
                else:
                    break
            ne = news[file_num][start: end]
    
            ne = 'ne_' + '_'.join(ne.split())
            ne = ''.join(['_' + i if i in punc else i for i in list(ne)])
    #        print(ne)
            prefix = news[file_num][0: start]
            suffix = news[file_num][end:]
            news[file_num] = prefix + ' ' + ne + ' ' + suffix
    
    
    with open(os.path.join(output_dir, str(index) + '.txt'), 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(news))
    endtime = datetime.datetime.now()
    print('Totol running for ', (endtime - starttime).seconds, ' seconds.')