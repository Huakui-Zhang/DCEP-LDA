'''
调用NeuroNER（https://github.com/Franck-Dernoncourt/NeuroNER）。
下载NeuroNer后，此文件应放于文件夹NeuroNER-master下。
输入为需要识别命名实体的数据集，每个文档单独作为一个文本文件。
输出为BRAT格式的文件，文件名为“数据集名.txt”
'''

from neuroner import neuromodel
import tensorflow as tf
import datetime
import os
import shutil
import warnings
warnings.filterwarnings("ignore")


#nn = neuromodel.NeuroNER(train_model=False, use_pretrained_model=True, \
#                         dataset_text_folder='./data/example_unannotated_texts', \
#                         pretrained_model_folder='./trained_models/conll_2003_en')
#
#nn.fit()
#nn.close()

dir_list = ['../20news_data/20news_raw','../reuters_data/reuters_raw']
for data_folder in dir_list:
    starttime = datetime.datetime.now()
    tf.reset_default_graph()
    print('*************************start', data_folder, '*******************************')
    output_folder = './output'
    # dataset_text_folder参数为输入文件夹，必须在该文件夹下新建“deploy”文件夹，在deploy文件夹下放入输入文档的的文本文件，可以放多个文件
    nn = neuromodel.NeuroNER(train_model=False, use_pretrained_model=True, \
                             dataset_text_folder=data_folder,\
                             pretrained_model_folder='./trained_models/conll_2003_en',\
                             output_folder=output_folder)
    nn.fit()
    nn.close()
    
    # 由于中间文件占据存储空间过大，删除多余的中间文件      
    dir_name = data_folder.split('/')[-1]
    output_data_folder = os.path.join(output_folder, dir_name)
    file_origin_path = os.path.join(output_data_folder,'000_deploy.txt')
    # 输出文件路径
    file_new_path = os.path.join(output_folder, dir_name+".txt")
    shutil.move(file_origin_path, file_new_path)
    shutil.rmtree(output_data_folder)
    endtime = datetime.datetime.now()
    print('Totol running for ', (endtime - starttime).seconds, ' seconds.')
    print('*************************end', data_folder, '*******************************')

