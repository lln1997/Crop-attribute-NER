import numpy as np


class HMM():
    def __init__(self, N, M):
        self.N = N  # 鐘舵�鏁�
        self.M = M  # 瑙傛祴鏁�

        self.A = np.zeros((N, N))  # 鐘舵�杞�Щ鐭╅樀
        self.B = np.zeros((N, M))  # 鍙戝皠鐭╅樀
        self.Pi = np.zeros(N)  # 鍒濆�姒傜巼鍚戦噺

    def train(self, word_lists, tag_lists, word2id, tag2id):
        """
        姹傜姸鎬佽浆绉荤煩闃�
        """
        for tag_list in tag_lists:
            for i in range(len(tag_list) - 1):
                cur_id = tag2id[tag_list[i]]
                next_id = tag2id[tag_list[i + 1]]
                self.A[cur_id][next_id] += 1
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)

        """
        姹傚彂灏勭煩闃�
        """
        for tag_list, word_list in zip(tag_lists, word_lists):
            for tag, word in zip(tag_list, word_list):
                tagid = tag2id[tag]
                wordid = word2id[word]
                self.B[tagid][wordid] += 1
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / np.sum(self.B, axis=1, keepdims=True)

        """
        姹傚垵濮嬫�鐜囩煩闃�
        """
        for tag_list in tag_lists:
            tagid = tag2id[tag_list[0]]
            self.Pi[tagid] += 1
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / np.sum(self.Pi)

    def test(self, word_lists, word2id, tag2id):
        pre_tag_lists = []
        n = len(word_lists)
        i = 1
        for word_list in word_lists:
            print(f"第{i}/{n}个句子正在解码")
            i += 1
            pre_tag_lists.append(self.decoding(word_list, word2id, tag2id))
        return pre_tag_lists

    def decoding(self, word_list, word2id, tag2id):
        A = np.log(self.A)
        B = np.log(self.B)
        Pi = np.log(self.Pi)

        seq_len = len(word_list)
        viterbi = np.zeros((self.N, seq_len))  # N=23
        backpointer = np.zeros((self.N, seq_len), dtype=int)

        start_wordid = word2id.get(word_list[0], None)
        Bt = B.T

        if start_wordid is None:
            bt = np.log(np.ones(self.N) / self.N)
        else:
            bt = Bt[start_wordid]  # 鑾峰彇鐭╅樀涓��
        viterbi[:, 0] = Pi + bt
        backpointer[:, 0] = -1

        for step in range(1, seq_len):
            wordid = word2id.get(word_list[step], None)
            if wordid is None:
                # 濡傛灉瀛椾笉鍐嶅瓧鍏搁噷锛屽垯鍋囪�鐘舵�鐨勬�鐜囧垎甯冩槸鍧囧寑鐨�
                bt = np.log(np.ones(self.N) / self.N)
            else:
                bt = Bt[wordid]  # 鍚﹀垯浠庤�娴嬫�鐜囩煩闃典腑鍙朾t
            for tag_id in range(len(tag2id)):
                maxpro = np.max(viterbi[:, step - 1] + A[:, tag_id], axis=0)
                maxid = np.argmax(viterbi[:, step - 1] + A[:, tag_id], axis=0)
                viterbi[tag_id, step] = maxpro + bt[tag_id]
                backpointer[tag_id, step] = maxid

        best_path_id = np.argmax(viterbi[:, seq_len - 1], axis=0)

        best_path = [best_path_id]
        for back in range(seq_len - 1, 0, -1):
            best_path_id = backpointer[best_path_id, back]
            best_path.append(best_path_id)

        id2tag = dict((id, tag) for tag, id in tag2id.items())
        taglist = [id2tag[tag] for tag in reversed(best_path)]
        return taglist
