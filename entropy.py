import jieba
import math
import time
import os
import re
import pdb
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
                with open(os.path.abspath(path), "r", encoding='ansi') as file:
                    filecontext = file.read()
                    filecontext = re.sub(r1, '', filecontext)
                    filecontext = filecontext.replace("\n", '')
                    filecontext = filecontext.replace(" ", '')
                    filecontext = filecontext.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
                    #seg_list = jieba.cut(filecontext, cut_all=True)
                    #corpus += seg_list
                    count += len(filecontext)
                    corpus.append(filecontext)
            elif os.path.isdir(path):
                TraversalFun.AllFiles(self, path)
        return corpus,count

# 词频统计，方便计算信息熵
def get_tf(tf_dic, words):

    for i in range(len(words)-1):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1

def get_bigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1

def get_trigram_tf(tf_dic, words):
    for i in range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1

def cal_unigram(corpus,count):
    before = time.time()
    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    for line in corpus:
        for x in jieba.cut(line):
            split_words.append(x)
            words_len += 1
        get_tf(words_tf, split_words)
        split_words = []
        line_count += 1

    print("语料库字数:", count)
    print("分词个数:", words_len)
    print("平均词长:", round(count / words_len, 5))
    entropy = []
    for uni_word in words_tf.items():
        entropy.append(-(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2))
    print("基于词的一元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")
    after = time.time()
    print("运行时间:", round(after - before, 5), "s")

def cal_bigram(corpus, count):
    before = time.time()
    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    bigram_tf = {}

    for line in corpus:
        for x in jieba.cut(line):
            split_words.append(x)
            words_len += 1

        get_tf(words_tf, split_words)
        get_bigram_tf(bigram_tf, split_words)

        split_words = []
        line_count += 1

    print("语料库字数:", count)
    print("分词个数:", words_len)
    print("平均词长:", round(count / words_len, 5))

    bigram_len = sum([dic[1] for dic in bigram_tf.items()])
    print("二元模型长度:", bigram_len)

    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    print("基于词的二元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")

    after = time.time()
    print("运行时间:", round(after - before, 5), "s")

def cal_trigram(corpus,count):
    before = time.time
    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    trigram_tf = {}

    for line in corpus:
        for x in jieba.cut(line):
            split_words.append(x)
            words_len += 1

        get_bigram_tf(words_tf, split_words)
        get_trigram_tf(trigram_tf, split_words)

        split_words = []
        line_count += 1

    print("语料库字数:", count)
    print("分词个数:", words_len)
    print("平均词长:", round(count / words_len, 5))

    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    print("三元模型长度:", trigram_len)

    entropy = []
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len  # 计算联合概率p(x,y)
        cp_xy = tri_word[1] / words_tf[tri_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
    print("基于词的三元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")

    after = time.time()
    print("运行时间:", round(after - before , 5), "s")

def cal_unigram_ch(corpus,count):
    before = time.time()
    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    for line in corpus:
        # for x in line:
        #     split_words.append(x)
        #     words_len += 1
        split_words = list(line)
        # words_len += len(words_len)
        get_tf(words_tf, split_words)
        split_words = []
        line_count += 1
    words_len = count
    print("语料库字数:", count)
    # print("分词个数:", words_len)
    # print("平均词长:", round(count / words_len, 5))
    entropy = []
    for uni_word in words_tf.items():
        entropy.append(-(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2))
    print("基于字的一元模型的中文信息熵为:", round(sum(entropy), 5), "比特/字")
    after = time.time()
    print("运行时间:", round(after - before, 5), "s")

def cal_bigram_ch(corpus, count):
    before = time.time()
    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    bigram_tf = {}

    for line in corpus:
        # for x in jieba.cut(line):
        #     split_words.append(x)
        #     words_len += 1
        split_words = list(line)
        get_tf(words_tf, split_words)
        get_bigram_tf(bigram_tf, split_words)

        split_words = []
        line_count += 1

    print("语料库字数:", count)
    # print("分词个数:", words_len)
    # print("平均词长:", round(count / words_len, 5))

    bigram_len = sum([dic[1] for dic in bigram_tf.items()])
    print("二元模型长度:", bigram_len)

    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    print("基于字的二元模型的中文信息熵为:", round(sum(entropy), 5), "比特/字")

    after = time.time()
    print("运行时间:", round(after - before, 5), "s")

def cal_trigram_ch(corpus,count):
    before = time.time()
    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    trigram_tf = {}

    for line in corpus:
        # for x in jieba.cut(line):
        #     split_words.append(x)
        #     words_len += 1
        split_words = list(line)
        get_bigram_tf(words_tf, split_words)
        get_trigram_tf(trigram_tf, split_words)

        split_words = []
        line_count += 1

    print("语料库字数:", count)
    # print("分词个数:", words_len)
    # print("平均词长:", round(count / words_len, 5))

    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    print("三元模型长度:", trigram_len)

    entropy = []
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len  # 计算联合概率p(x,y,z)
        cp_xy = tri_word[1] / words_tf[tri_word[0][0]]  # 计算条件概率p(x|y,z)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
    print("基于字的三元模型的中文信息熵为:", round(sum(entropy), 5), "比特/字")

    after = time.time()
    print("运行时间:", round(after - before , 5), "s")


if __name__ == '__main__':
    tra = TraversalFun("./datasets")
    corpus,count = tra.TraversalDir()
    # cal_unigram(corpus, count)
    # cal_bigram(corpus,count)
    # cal_trigram(corpus,count)
    # cal_unigram_ch(corpus, count)
    # cal_bigram_ch(corpus,count)
    cal_trigram_ch(corpus,count)

'''
语料库字数: 7420081
分词个数: 4430767
平均词长: 1.67467
基于词的一元模型的中文信息熵为: 12.01312 比特/词
运行时间: 26.30019 s
语料库字数: 7420081
分词个数: 4430767
平均词长: 1.67467
二元模型长度: 4430751
基于词的二元模型的中文信息熵为: 6.8915 比特/词
运行时间: 29.84001 s
语料库字数: 7420081
分词个数: 4430767
平均词长: 1.67467
三元模型长度: 4430735
基于词的三元模型的中文信息熵为: 2.41661 比特/词

语料库字数: 7420081
基于字的一元模型的中文信息熵为: 9.50211 比特/字
运行时间: 1.14236 s
语料库字数: 7420081
二元模型长度: 7420065
基于字的二元模型的中文信息熵为: 6.68521 比特/字
运行时间: 5.52439 s
语料库字数: 7420081
三元模型长度: 7420049
基于字的三元模型的中文信息熵为: 3.94294 比特/字
运行时间: 15.19367 s
'''