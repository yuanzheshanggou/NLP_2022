import os
import jieba
import warnings
import gensim.models as w2v
import re
import pdb

from sqlalchemy import true

class TraversalFun():

    # 1 初始化
    def __init__(self, rootDir, stop_words):
        self.rootDir = rootDir
        self.stop_words = stop_words

    def TraversalDir(self):
        return self.getCorpus()

    def getCorpus(self):
        rootDir = self.rootDir
        stop_words = self.stop_words
        corpus = []
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
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
                    # for word in stop_words:
                    #     filecontext = filecontext.replace(word, "")
                    filecontext = filecontext.split('。') 
                    #seg_list = jieba.cut(filecontext, cut_all=True)
                    #corpus += seg_list
                    count += len(filecontext)
                    corpus.extend(filecontext)
        return corpus,count

def get_words(filename):
    with open(filename, "r", encoding='gbk', errors="ignore") as f:
        words = [word.strip() for word in f.readlines()]
    return words

def add_words(words):
    for word in words:
        jieba.add_word(word)
    return

def train_model(sentences):
    model = w2v.Word2Vec(sentences=sentences, min_count=5, size=300, window=5, sg=1, iter=10)
    model.save('./CBOW.model')  # 保存模型
    model.wv.save_word2vec_format("./CBOW_300vec.txt", binary=False)

def find_relation(model, a, b, c):
    d, _ = model.wv.most_similar(positive=[c, b], negative=[a])[0]
    print (c,d)


def train():
    stop_words = get_words("./人物武功门派和停词/stop_words.txt"); stop_words.remove('。')
    menpai = get_words("./人物武功门派和停词/金庸小说全门派.txt")
    renwu = get_words("./人物武功门派和停词/金庸小说全人物.txt")
    wugong = get_words("./人物武功门派和停词/金庸小说全武功.txt")
    add_words(menpai); add_words(renwu); add_words(wugong)    
    tra = TraversalFun("./datasets", stop_words)
    corpus,count = tra.TraversalDir()
    sentences = [jieba.lcut(sentence) for sentence in corpus]
    
    
    model = train_model(sentences)


def test():
    model =  w2v.Word2Vec.load("./CBOW.model")
    renwu = ["杨过", "杨康", "郭靖", "段誉", "张无忌", "令狐冲", "风清扬", "林平之", "狄云", "胡斐", "小龙女", "穆念慈", "黄蓉", "王语嫣", "赵敏", "阿朱", "岳灵珊", "李莫愁", "陆无双", "程英"]
    diming = ["桃花岛", "襄阳", "昆仑", "西夏", "大理"]
    wugong = ["凌波微步", "六脉神剑", "黯然销魂掌", "降龙十八掌", "一阳指", "北冥神功", "化功大法", "吸星大法"]
    for renwu in renwu:
        print(renwu)
        print(model.wv.most_similar(renwu, topn=10))
    for item in diming:
        print(item)
        print(model.wv.most_similar(item, topn=10))
    for item in wugong:
        print(item)
        print(model.wv.most_similar(item, topn=20))

    find_relation(model, "郭靖", "降龙十八掌", "鲁有脚")



if __name__ == "__main__":
    train()
    test()
