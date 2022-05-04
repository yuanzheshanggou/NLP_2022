from gensim.models import LdaModel
from gensim import corpora
import jieba
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split as ts

import os
import re
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class TraversalFun():

    # 1 初始化
    def __init__(self, rootDir):
        self.rootDir = rootDir

    def TraversalDir(self):
        return self.getCorpus(self.rootDir)

    def getCorpus(self, rootDir):
        corpus = []
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
        listdir = os.listdir(rootDir)
        count=0
        for file in listdir:
            path  = os.path.join(rootDir, file)
            if os.path.isfile(path):
                with open(os.path.abspath(path), "r", encoding='gbk', errors="ignore") as file:
                    filecontext = file.read()
                    filecontext = re.sub(r1, '', filecontext)
                    filecontext = filecontext.replace("\n", '')
                    filecontext = filecontext.replace(" ", '')
                    filecontext = filecontext.replace("\u3000", '')
                    filecontext = filecontext.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
                    filecontext = filecontext.replace("本书来自免费小说下载站\n更多更新免费电子书请关注", '')
                    #seg_list = jieba.cut(filecontext, cut_all=True)
                    #corpus += seg_list
                    count += len(filecontext)
                    corpus.append(filecontext)
        return corpus,count

def construct_dataset(corpus, paragraphs_per_book=15, words_per_paragraph=500):
    paragraphs, bookid = random_select(corpus, paragraphs_per_book, words_per_paragraph)
    
    dictionary = corpora.Dictionary(paragraphs)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    dictionary.compactify()
    corpus = [dictionary.doc2bow(text) for text in paragraphs] 

    # Set training parameters.
    num_topics = 30
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = 100  # Don't evaluate model perplexity, takes too much time.

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    
    topics_ = lda.get_document_topics(corpus, minimum_probability=0)
    topics = [[t[1] for t in topic] for topic in topics_]
    topics = np.array(topics)
    bookid = np.array(bookid)
    return topics, bookid
    
    
def random_select(corpus, paragraphs_per_book, words_per_paragraph):
    paragraphs = list()
    bookid = list()
    for i, text in enumerate(corpus):
        stopwords = get_stopwords()
        text = [p for p in jieba.cut(text) if p not in stopwords]
        for _ in range(paragraphs_per_book):
            p = np.random.randint(0, len(text)-words_per_paragraph)
            paragraphs.append(text[p:p+words_per_paragraph])
            bookid.append(i)
    return paragraphs, bookid

        
def get_stopwords():
    with open("./cn_stopwords.txt", "r") as f:
        stopwords = f.readlines()

    return [word.replace('\n', '') for word in stopwords]

def main():
    # 准备数据
    tra = TraversalFun("./datasets")
    corpus,count = tra.TraversalDir()
    data, target = construct_dataset(corpus, 20, 500)
    X_train,X_test,y_train,y_test = ts(data, target, test_size=0.3)
    # kernel = 'rbf'
    clf_rbf = svm.SVC(kernel='poly')
    clf_rbf.fit(X_train,y_train)
    score_rbf = clf_rbf.score(X_test,y_test)
    print("The score of rbf is : %f"%score_rbf)



 
if __name__ == '__main__':
    main()
