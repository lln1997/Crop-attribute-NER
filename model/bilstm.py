import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        # 初始化一个随机矩阵，长为字典大小，宽为字典中字的特征数，
        # 类实例化之后可以根据字典中元素的下标来查找元素对应的向量。输入下标0，输出就是embedding矩阵中第0行。
        self.embedding = nn.Embedding(vocab_size, emb_size)  # 此例（1794,128）
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)  # hidden_size指的是W的维数（第二维）

        self.lin = nn.Linear(2 * hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)  # [B(batch_size), L(最长句子的长度), emb_size(特征数)] 此例（64,178，128）

        packed = pack_padded_sequence(emb, lengths, batch_first=True)  # lengths每句的实际长度，这里输出降成二维的了
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.lin(rnn_out)  # [B, L, out_size]
        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)  # out_size最大值,返回的是下标

        return batch_tagids  # 二维矩阵，第一维测试集总数，第二维是每个测试集中每个字对应的最大值的标签下标
