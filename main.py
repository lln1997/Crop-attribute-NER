# coding:utf-8
from read_data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf, load_model
from evaluate import hmm_train_eval, crf_train_eval, bilstm_train_and_eval, Metrics


def main():
    print("读取数据")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
    # 训练评估HMM模型
    print("正在训练评估HMM模型...")
    hmm_pred = hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )
    hmm_model = load_model("./ckpts/hmm.pkl")
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)
    #
    # metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=True)
    # metrics.report_scores()
    # metrics.report_confusion_matrix()

    # 训练评估CRF模型
    # print("正在训练评估CRF模型...")
    # crf_pred = crf_train_eval(
    #     (train_word_lists, train_tag_lists),
    #     (test_word_lists, test_tag_lists)
    # )

    # 训练评估BI-LSTM模型
    # print("正在训练评估双向LSTM模型...")
    # #LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    # bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    # lstm_pred = bilstm_train_and_eval(
    #     (train_word_lists, train_tag_lists),
    #     (dev_word_lists, dev_tag_lists),
    #     (test_word_lists, test_tag_lists),
    #     bilstm_word2id, bilstm_tag2id,
    #     crf=False
    # )

    # print("正在训练评估Bi-LSTM+CRF模型...")
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    #crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    ## 还需要额外的一些数据处理
    #train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
        #train_word_lists, train_tag_lists
    #)
    #dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
        #dev_word_lists, dev_tag_lists
    #)
    #test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        #test_word_lists, test_tag_lists, test=True
    #)
    ## lstmcrf_pred = bilstm_train_and_eval(
    ##     (train_word_lists, train_tag_lists),
    ##     (dev_word_lists, dev_tag_lists),
    ##     (test_word_lists, test_tag_lists),
    ##     crf_word2id, crf_tag2id
    ## )
    #bilstm_model = load_model("./ckpts/bilstm_crf.pkl")
    #pred_tag_lists, test_tag_lists = bilstm_model.test(
        #test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=True)
    metrics.report_scores()
    metrics.report_confusion_matrix()


if __name__ == '__main__':
    main()
