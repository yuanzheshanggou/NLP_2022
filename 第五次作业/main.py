import jieba
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split as ts
import re
from gensim.models import Word2Vec
import torch
import torch.nn as nn
from model import Seq2Seq
from tqdm import trange
import os
import re
import logging
import pdb
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class TraversalFun():

    # 1 初始化
    def __init__(self, rootDir="../LDA/datasets/"):
        self.rootDir = rootDir

    def TraversalDir(self):
        return self.getCorpus(self.rootDir)
    
    def get_sentences(self, bookname, rootDir="../LDA/datasets/"):
        if os.path.exists(os.path.join(rootDir, bookname)):
            corpus = self._getCorpus(bookname, rootDir)
            return self._get_sentences(corpus)

    def _getCorpora(self, rootDir):
        '''
        corpus:长度为16的list,每个元素为储存书内容的字典
        count:书的数量
        '''
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
                    # filecontext = filecontext.replace("\n", '')
                    filecontext = filecontext.replace(" ", '')
                    filecontext = filecontext.replace("\u3000", '')
                    filecontext = filecontext.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
                    filecontext = filecontext.replace("本书来自免费小说下载站\n更多更新免费电子书请关注", '')
                    #seg_list = jieba.cut(filecontext, cut_all=True)
                    #corpus += seg_list
                    count += len(filecontext)
                    corpus.append(filecontext)
        return corpus,count
    
    def _getCorpus(self, bookname, rootDir="../LDA/datasets/"):
        bookpath = os.path.join(rootDir, bookname)
        r1 = u'[a-zA-Z0-9’"#$%&\'()*+,-./:：;<=>?@，★、…【】《》“”‘’[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
        if os.path.isfile(bookpath):
            with open(os.path.abspath(bookpath), "r", encoding='gbk', errors="ignore") as file:
                filecontext = file.read()
                filecontext = re.sub(r1, '', filecontext)
                filecontext = filecontext.replace("\n", '')
                filecontext = filecontext.replace(" ", '')
                filecontext = filecontext.replace("\u3000", '')
                filecontext = filecontext.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
                filecontext = filecontext.replace("本书来自免费小说下载站\n更多更新免费电子书请关注", '')
                #seg_list = jieba.cut(filecontext, cut_all=True)
                #corpus += seg_list                
        return filecontext

    def _get_sentences(self, corpus, length=30):
        '''
        sentences:列表，每个元素为句子
        '''
        sybl = "。|？|！|……"
        sentences_tmp=re.split(sybl, corpus)
        sentences = list()
        for sentence in sentences_tmp:
            sentence_split = list(jieba.cut(sentence, cut_all=False))
            if len(sentence_split) < 5:
                continue
            sentence_split.append("END")
            sentences.append(sentence_split)
        print("已获得句子与分词")
        return sentences
    
     

class W2V(object):
    def __init__(self, sentences, sg=0, embed_size=300, min_count=1, window=10, iter=20):
        super().__init__()
        if not os.path.exists('w2v.model'):
            self.model = Word2Vec(sentences=sentences, sg=sg, size=embed_size, min_count=min_count, window=window, iter=iter)
            self.model.save('w2v.model')
        else:
            self.model = Word2Vec.load('w2v.model')
    def get_model(self):
        return self.model

def train(sentences, w2v_model, seq2seq_model, embed_size=300, epochs=100, end_num=10):
    optimizer = torch.optim.SGD(params=seq2seq_model.parameters(), lr=0.01)
    for epoch_id in range(epochs):
        for idx in trange(0, len(sentences) // end_num - 1):
            seq = []
            for k in range(end_num):
                seq += sentences[idx + k]
            target = []
            for k in range(end_num):
                target += sentences[idx + end_num + k]
            input_seq = torch.zeros(len(seq), embed_size)
            for k in range(len(seq)):
                input_seq[k] = torch.tensor(w2v_model.wv[seq[k]])
            target_seq = torch.zeros(len(target), embed_size)
            for k in range(len(target)):
                target_seq[k] = torch.tensor(w2v_model.wv[target[k]])
            all_seq = torch.cat((input_seq, target_seq), dim=0)
            optimizer.zero_grad()
            out_res = seq2seq_model(all_seq[:-1])
            f1 = ((out_res[-target_seq.shape[0]:] ** 2).sum(dim=1)) ** 0.5
            f2 = ((target_seq.cuda() ** 2).sum(dim=1)) ** 0.5
            loss = (1 - (out_res[-target_seq.shape[0]:] * target_seq.cuda()).sum(dim=1) / f1 / f2).mean()
            loss.backward()
            optimizer.step()
        if idx % (epochs-1) == 0:
            print("loss: ", loss.item(), " in epoch ", epoch_id, " res: ",out_res[-target_seq.shape[0]:].max(dim=1).indices, target_seq.max(dim=1).indices)
        
    torch.save(seq2seq_model.state_dict(), "model/" + "Seq2Seq.pth.tar")
        
def test(sentences, w2v_model, seq2seq_model, embed_size=300):
    seqs = list()
    for s in sentences:
        seqs += s
    input_seq = torch.zeros(len(seqs), embed_size).cuda()
    result = ""
    with torch.no_grad():
        for k in range(len(seqs)):
            try:
                input_seq[k] = torch.tensor(w2v_model.wv[seqs[k]])
            except:
                continue
        end_num = 0
        length = 0
        while end_num < 10 and length < 200:
            print("length: ", length)
            out_res = seq2seq_model(input_seq)[-1:]
            key_value = w2v_model.wv.most_similar(positive=np.array(out_res.cpu()), topn=20)
            key=key_value[0][0]
            if key == "END":
                result += "。"
                end_num += 1
            else:
                result += key
            length += 1
            input_seq = torch.cat((input_seq, out_res), dim=0)
    print(result)

def init():
    stop_words = get_words("./人物武功门派和停词/stop_words.txt"); stop_words.remove('。')
    menpai = get_words("./人物武功门派和停词/金庸小说全门派.txt")
    renwu = get_words("./人物武功门派和停词/金庸小说全人物.txt")
    wugong = get_words("./人物武功门派和停词/金庸小说全武功.txt")
    add_words(menpai); add_words(renwu); add_words(wugong)    

def add_words(words):
    for word in words:
        jieba.add_word(word)
    return

def get_words(filename):
    with open(filename, "r", encoding='gbk', errors="ignore") as f:
        words = [word.strip() for word in f.readlines()]
    return words



def main(unse_checkpoint=1):
    init()
    tra = TraversalFun()
    sentences = tra.get_sentences("天龙八部.txt")
    embed_size=300
    w2v_model = W2V(sentences, embed_size=embed_size).get_model()
    seq2seq_model = nn.DataParallel(Seq2Seq(embed_size)).cuda()
    
    if unse_checkpoint and os.path.exists("./model/Seq2Seq.pth.tar"):
        checkpoint = torch.load("./model/Seq2Seq.pth.tar")
        seq2seq_model.load_state_dict(checkpoint)
    else:
        train(sentences, w2v_model, seq2seq_model, embed_size=embed_size)

    test(tra.get_sentences("test.txt", "./"), w2v_model, seq2seq_model, embed_size=embed_size)






if __name__ == "__main__":
    main(1)
