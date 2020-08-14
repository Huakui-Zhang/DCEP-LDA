# -*- coding: utf-8 -*-
"""
处理维基百科，输入为wikiextractor（https://github.com/attardi/wikiextractor）处理后的维基百科文件，
将维基百科每个文档写入一个单独的文件中，用于后续NER处理。由于维基百科过大，因此分散再不同文件夹中进行NER识别。
"""
import os
import json

num_wiki_file = 130
# 一篇文档最大字符数，由于NeuroNER最大只能处理不超过100万字符的文档
max_doc_length = 950000
# 一个文件夹中所有文件的最大字符数
max_dir_length = 5000000

raw_data_dir = './enwiki-20190301/raw'
result_data_dir = './enwiki-20190301/preprocess_data'
seperate_data_dir = './enwiki-20190301/seperate_preprocess_data'

start_index = 0
length = 0
dir_num = 0
text_cache = []

for i in range(0, num_wiki_file):
    raw_file_num = '%02d'%i
    print('processing file', raw_file_num)
    raw_file_name = 'wiki_' + raw_file_num
    raw_file = os.path.join(raw_data_dir, raw_file_name)
    raw_file_num = i

    with open(raw_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    lines = [line.encode('utf8')[3:].decode('utf8') if line.startswith(u'\ufeff') else line for line in lines]
        
    lines = [json.loads(line)['text'] for line in lines]
    lines = [' '.join(line.split()) for line in lines]
    lines = [line[0: max_doc_length] if len(line) > max_doc_length else line for line in lines]
    
    if not os.path.exists(result_data_dir):
        os.makedirs(result_data_dir)
    with open(os.path.join(result_data_dir, str(i) + '.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    for i in range(len(lines)):
        data_dir = seperate_data_dir + '/' + str(dir_num) + '/deploy'
        aggregate_data_dir = seperate_data_dir + '/' + str(dir_num)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        length += len(lines[i]) + 1
        # 没达到最大长度
        if length < max_dir_length:
            # 是文件最后一行
            if i == len(lines) - 1:
                # 不是最后一个文件
                if raw_file_num != num_wiki_file - 1:
                    text_cache.append(lines[i])
                # 是最后一个文件
                else:
                    for index in range(len(text_cache)):
                        with open(os.path.join(data_dir, str(index) + '.txt'), 'w', encoding='utf-8') as f:
                            f.write(text_cache[index])
                    with open(os.path.join(aggregate_data_dir, '0.txt'), 'w', encoding='utf-8') as f:
                            f.write('\n'.join(text_cache))
            # 不是文件最后一行
            else:
                text_cache.append(lines[i])
                
    
        else:
            # 是文件最后一行
            if i == len(lines) - 1:
                for index in range(len(text_cache)):
                    with open(os.path.join(data_dir, str(index) + '.txt'), 'w', encoding='utf-8') as f:
                        f.write(text_cache[index])
                with open(os.path.join(aggregate_data_dir, '0.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(text_cache))
                    
                dir_num += 1
                length = len(lines[i]) + 1
                text_cache = [lines[i]]
                # 是最后一个文件
                if raw_file_num == num_wiki_file - 1:
                    data_dir = result_data_dir + str(dir_num) + '/deploy'
                    aggregate_data_dir = seperate_data_dir + str(dir_num)
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
                    for index in range(len(text_cache)):
                        with open(os.path.join(data_dir, str(index) + '.txt'), 'w', encoding='utf-8') as f:
                            f.write(text_cache[index])
                    with open(os.path.join(aggregate_data_dir, '0.txt'), 'w', encoding='utf-8') as f:
                        f.write('\n'.join(text_cache))
            
            # 不是文件最后一行
            else:
                for index in range(len(text_cache)):
                    with open(os.path.join(data_dir, str(index) + '.txt'), 'w', encoding='utf-8') as f:
                        f.write(text_cache[index])
                with open(os.path.join(aggregate_data_dir, '0.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(text_cache))
                    
                dir_num += 1
                length = len(lines[i]) + 1
                text_cache = [lines[i]]