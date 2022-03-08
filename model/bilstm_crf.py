from itertools import zip_longest
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from .util import tensorized, sort_by_lengths, cal_loss, cal_lstm_crf_loss
from .config import TrainingConfig, LSTMConfig
from .bilstm import BiLSTM
import os


class BILSTM_Model(object):
    def __init__(self, vocab_size, out_size, crf=True):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            crf选择是否添加CRF层"""
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型参数
        self.emb_size = LSTMConfig.emb_size
        self.hidden_size = LSTMConfig.hidden_size
        self.crf = crf
        # 根据是否添加crf初始化不同的模型 选择不一样的损失计算函数
        if not crf:
            self.model = BiLSTM(vocab_size, self.emb_size,
                                self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_loss
        else:
            self.model = BiLSTM_CRF(vocab_size, self.emb_size,
                                    self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_lstm_crf_loss

        # 加载训练参数：
        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        # 对数据集按照长度进行排序(该例从长到短)
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
            dev_word_lists, dev_tag_lists)
        t = 0
        B = self.batch_size  # 每批向量的大小
        for e in range(1, self.epoches + 1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind + B]  # B个
                batch_tags = tag_lists[ind:ind + B]

                losses += self.train_step(batch_sents,
                                          batch_tags, word2id, tag2id)

                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(
                dev_word_lists, dev_tag_lists, word2id, tag2id)
            if val_loss < self._best_val_loss:
                print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss
                t = 0
            elif val_loss > self._best_val_loss:
                t += 1
            if t >= 5:
                print("模型已收敛")
                break
            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()  # 添加归一化和dropout处理，在训练开始前加上
        self.step += 1
        # 准备数据
        tensorized_sents, lengths = tensorized(batch_sents, word2id)  # 此例（64,178）
        tensorized_sents = tensorized_sents.to(self.device)  # 把矩阵放到gpu上运行
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        scores = self.model(tensorized_sents, lengths)  # 把model当做函数使用就会自动调用forward函数

        # 计算损失 更新参数
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新参数

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():  # 不需计算梯度和反向传播
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind + self.batch_size]
                batch_tags = dev_tag_lists[ind:ind + self.batch_size]
                tensorized_sents, lengths = tensorized(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            # if val_loss < self._best_val_loss:
            #     print("保存模型...")
            #     self.best_model = deepcopy(self.model)
            #     self._best_val_loss = val_loss

            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """返回最佳模型在测试集上的预测结果"""
        pred_res = []
        batch_size = 20
        bs = [i for i in range(0, len(word_lists), batch_size)]
        if bs[-1] < len(word_lists):
            bs.append(len(word_lists))
        for i in range(len(bs) - 1):
            # 准备数据
            word_lists[bs[i]:bs[i + 1]], tag_lists[bs[i]:bs[i + 1]], indices = sort_by_lengths(
                word_lists[bs[i]:bs[i + 1]], tag_lists[bs[i]:bs[i + 1]])
            # 让word_lists变成等长句子矩阵
            tensorized_sents, lengths = tensorized(word_lists[bs[i]:bs[i + 1]], word2id)
            tensorized_sents = tensorized_sents.to(self.device)

            self.best_model.eval()
            # 预测结果
            with torch.no_grad():
                # 调用的模型里的test，返回的是二维矩阵，第一维测试集总数，第二维是每个测试集中的字对应的最大值的标签的下标
                batch_tagids = self.best_model.test(
                    tensorized_sents, lengths, tag2id)

            # 将id转化为标注（'O','B-NAME'）之类的
            pred_tag_lists = []
            id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
            for k, ids in enumerate(batch_tagids):
                tag_list = []
                if self.crf:
                    for j in range(lengths[k] - 1):  # crf解码过程中，end被舍弃
                        tag_list.append(id2tag[ids[j].item()])
                else:
                    for j in range(lengths[k]):
                        tag_list.append(id2tag[ids[j].item()])
                pred_tag_lists.append(tag_list)

            # indices存有根据长度排序后的索引映射的信息
            # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
            # 索引为2的元素映射到新的索引是1...
            # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
            # list(enumerate(indices)) 假如indices=[1,2,0]，返回的是[(0,1),(1,2),(2,0)]
            # 上面的例子经过下面的sorted返回的是[(2,0),(0,1),(1,2)]
            ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
            indices, _ = list(zip(*ind_maps))
            pred_res[bs[i]:bs[i + 1]] = [pred_tag_lists[i] for i in indices]
            tag_lists[bs[i]:bs[i + 1]] = [tag_lists[bs[i]:bs[i + 1]][j] for j in indices]

        return pred_res, tag_lists


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)

        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1 / out_size)
        # self.transition.data1.zero_()

    def forward(self, sents_tensor, lengths):
        # [B, L, out_size] 此例为[64,178,32] ,等于发射矩阵
        emission = self.bilstm(sents_tensor, lengths)  # 调用BiLSTM的forward函数

        # 计算CRF scores, 这个scores大小为[B, L, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        batch_size, max_len, out_size = emission.size()

        # c=emission.unsqueeze(2)#unsqueeze在第二维上加上一维为[64,178,1,32]
        # d=c.expand(-1, -1, out_size, -1) #单纯将第三维扩充到out_size其余的不动 [64,178,32,32]
        # e=self.transition.unsqueeze(0)#[1,32,32]
        # f=d+e #每个样本的每个字都加上转移矩阵

        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)
        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        """使用维特比算法进行解码,test_sents_tensor此例为[477,168]"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)  # [477,168,32,32]
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()  # B:477 L:168 T:32
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)  # [477,168,32]
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)  # [477,168,32]
        lengths = torch.LongTensor(lengths).to(device)  # [477]
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()  # 返回的是大于lengths大于step的值的个数，每次处理的批数
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                # 一个逗号表示一维
                viterbi[:batch_size_t, step,
                :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                # aaa=viterbi[:batch_size_t, step-1, :] #[477,32]
                # aco=viterbi[:batch_size_t, step-1, :].unsqueeze(2) #[477,32,1]
                # bco=crf_scores[:batch_size_t, step, :, :] #[477,32,32]

                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step - 1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],  # [B, T, T]
                    dim=1
                )  # dim为多少就是表示不要该维了，转化成其余维度最大时在该维的下标
                viterbi[:batch_size_t, step, :] = max_scores  # max_score [477,32]
                backpointer[:batch_size_t, step, :] = prev_tags  # pre_tags [477,32]

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T] [477,5376]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L - 1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L - 1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids
