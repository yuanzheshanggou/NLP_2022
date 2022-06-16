import torch
import torch.nn as nn

class Seq2SeqEncoder(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.linear = nn.Linear(embed_size, embed_size)
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        X = self.linear(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state

class Seq2SeqDecoder(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, embed_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        output, state = self.rnn(X, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state

class Seq2Seq(nn.Module):
    def __init__(self, onehot_num):
        super(Seq2Seq, self).__init__()
        onehot_size = onehot_num
        embedding_size = 512
        n_layer = 2
        self.lstm = nn.LSTM(embedding_size, embedding_size, n_layer, batch_first=True)# 编码
        self.encode = torch.nn.Sequential(nn.Linear(onehot_size, embedding_size),nn.Dropout(0.5),nn.ReLU())# 解码
        self.decode = torch.nn.Sequential(nn.Linear(embedding_size, onehot_size),nn.Dropout(0.5))

    def forward(self, x):# 入
        em = self.encode(x).unsqueeze(dim=1)# 出
        out, (h, c) = self.lstm(em)
        res = self.decode(out.squeeze(1))
        return res
