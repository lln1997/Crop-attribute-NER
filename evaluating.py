from collections import Counter

from utils import flatten_lists


class Metrics(object):
    """鐢ㄤ簬璇勪环妯″瀷锛岃�绠楁瘡涓�爣绛剧殑绮剧‘鐜囷紝鍙�洖鐜囷紝F1鍒嗘暟"""

    # golden_tags鏄�祴璇曟暟鎹�殑tag
    def __init__(self, golden_tags, predict_tags, remove_O=False):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # 灏哋鏍囪�绉婚櫎锛屽彧鍏冲績瀹炰綋鏍囪�
            self._remove_Otags()

        # 杈呭姪璁＄畻鐨勫彉閲�        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)  # 璁＄畻tag鐨勬�鏁拌繑鍥炴槸瀛楀吀渚嬪�{'O':456}
        self.golden_tags_counter = Counter(self.golden_tags)

        # 璁＄畻绮剧‘鐜�        self.precision_scores = self.cal_precision()

        # 璁＄畻鍙�洖鐜�        self.recall_scores = self.cal_recall()

        # 璁＄畻F1鍒嗘暟
        self.f1_scores = self.cal_f1()

    def cal_precision(self):

        precision_scores = {}
        for tag in self.tagset:
            if tag not in self.predict_tags_counter:
                precision_scores[tag] = 0
            else:
                precision_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                    self.predict_tags_counter[tag]

        return precision_scores

    def cal_recall(self):

        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.golden_tags_counter[tag]
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2 * p * r / (p + r + 1e-10)  # 鍔犱笂涓�釜鐗瑰埆灏忕殑鏁帮紝闃叉�鍒嗘瘝涓�
        return f1_scores

    def report_scores(self):
        """灏嗙粨鏋滅敤琛ㄦ牸鐨勫舰寮忔墦鍗板嚭鏉ワ紝鍍忚繖涓�牱瀛愶細

                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634

          avg/total      0.779     0.764     0.770      6178
        """
        # 鎵撳嵃琛ㄥご
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 鎵撳嵃姣忎釜鏍囩�鐨�绮剧‘鐜囥�鍙�洖鐜囥�f1鍒嗘暟
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ))

        # 璁＄畻骞舵墦鍗板钩鍧囧�
        avg_metrics = self._cal_weighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ))

    def count_correct_tags(self):
        """璁＄畻姣忕�鏍囩�棰勬祴姝ｇ‘鐨勪釜鏁�瀵瑰簲绮剧‘鐜囥�鍙�洖鐜囪�绠楀叕寮忎笂鐨則p)锛岀敤浜庡悗闈㈢簿纭�巼浠ュ強鍙�洖鐜囩殑璁＄畻"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):  # zip鎵撳寘鎴愬厓缁�            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 璁＄畻weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def _remove_Otags(self):

        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O']

        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]

        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        print("鍘熸�鏍囪�鏁颁负{}锛岀Щ闄や簡{}涓狾鏍囪�锛屽崰姣攞:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_confusion_matrix(self):
        """璁＄畻娣锋穯鐭╅樀"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 鍒濆�鍖栨贩娣嗙煩闃�matrix[i][j]琛ㄧず绗琲涓猼ag琚�ā鍨嬮�娴嬫垚绗琷涓猼ag鐨勬�鏁�        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 閬嶅巻tags鍒楄〃
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 鏈夋瀬灏戞暟鏍囪�娌℃湁鍑虹幇鍦╣olden_tags锛屼絾鍑虹幇鍦╬redict_tags锛岃烦杩囪繖浜涙爣璁�                continue

        # 杈撳嚭鐭╅樀
        row_format_ = ('{:>7} ' * (tags_size + 1))
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))
