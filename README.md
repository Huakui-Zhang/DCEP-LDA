# NEP-LDA
Implementation of paper Incorporating Concept Information into Term Weighting Schemes for Topic Models

## 数据集：
[20 Newsgroups](http://qwone.com/∼jason/20Newsgroups)
<br>
[Reuters corpus](http://www.nltk.org/book/ch02.html)

## 评价指标：
[C_v](https://dl.acm.org/citation.cfm?id=2685324)

## 文件运行步骤：
1. get_20news.py 用于获取20NewGroups数据集，并生成用于NeuroNER识别的数据集文件
2. get_wiki 生成用于NeuroNER识别的维基百科文件
3. NeurNER.py 使用NeuroNER进行命名实体识别
4. NER_process.py 用于处理NeuroNER产生的文件
5. NER_wiki_process.py 用于处理NeuroNER产生的文件
6. LDA.py LDA的实现以及主题连贯性测试
7. NEP-LDA.py "Document Dependent Named Entity Promoting"词权重的实现以及主题连贯性测试
8. CEP-LDA.py "Concept Based Entity Promoting"词权重的实现以及主题连贯性测试
9. DCEP-LDA.py "Duplicated Concept Based Entity Promoting"词权重的实现以及主题连贯性测试

## 实现经验总结：
一、[NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER) 使用注意事项：
1. 在使用NeuroNER时，要注意dataset_text_folder参数并不是放需要NER识别的文档所在的文件夹路径。我们需要在dataset_text_folder则个文件夹下新建一个deploy文件夹，然后在这个文件夹下放需要识别的文档。
2. 并不是把整个数据集都写入一个文件来进行NER识别，而是每个文档单独一个文件，多个文件可以放在同一个deploy文件夹下。
3. 需要识别的文档字符长度不能超过100万，否则会报错。
4. deploy文件夹下的文件越多，运行NeuroNER时，需要的内存越多，超出内训时可能会出现memory error错误，在12G内存的电脑上，deploy文件夹下的所有文档加起来的字符数最优大概是500万字符。数据集过大或者需要对整个维基百科识别时，可以把所有文档文件分散在多个文件夹中，当然每个文件夹下都要有deploy文件夹，这才是文档该放的位置，然后使用for循环，dataset_text_folder在每次for循环中设置不同文件夹路径。
5. 同一个deploy文件夹下的所有文档经过NeuroNER识别后只输出一个文件，名称默认为 000_deploy.txt。

二、Gensim的ldamulticore模型在windows10系统下的spyder中无法进行多核训练，在Jupyter中可以使用多核，在PyCharm中未尝试。

三、在使用Gensim中coherencemodel中的C_v主题连贯性评价指标时的注意事项
1. 如果外部语料库（维基百科）中没有一个词语，但在一个主题中有这个词语，那么这个主题的连贯性输出为nan。因此在使用C_v计算主题连贯性前，需要先确定主题中的每个词语是不是都在外部语料库中出现过，如果没有出现一个词语，我的做法是把这个词语在这个主题中删除。
2. coherencemodel中C_v评价指标的函数中的参数texts需要传入处理成list of list of tokens的外部语料库，但由于维基百科数据过大，电脑内存小的话，无法把所有维基百科处理成上述格式再调用该评价指标，我的做法是通过修改Gensim库中的部分文件，将texts参数赋值为维基百科所在的文件夹，使coherencemodel再一个for循环中处理该文件夹中的每一个小的维基百科文件，而不用将这些小的文件合并成一个大文件再传入。主要修改的文件为：C:\ProgramData\Anaconda3\Lib\site-packages\gensim\topic_coherence\text_analysis.py，WindowedTextsAnalyzer类，_iter_texts方法，该方法中的texts参数即coherencemodel中的text参数。我上传了修改后的3个文件在code about coherence文件夹中，分别是C:\ProgramData\Anaconda3\Lib\site-packages\gensim\models中的coherencemodel_modify_line_36_39.py，C:\ProgramData\Anaconda3\Lib\site-packages\gensim\topic_coherence中的probability_estimation_modify_line_12.py和text_analysis_modify_line_301.py
