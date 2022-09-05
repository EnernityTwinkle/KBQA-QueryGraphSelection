import sys
import os
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


# 从已经拍好序的候选文件中选出top_n构建重排序数据
def get_top_n(file_name, top_n, qid_pos = -2):
    top_n_file_name = file_name.replace('.txt', '_top' + str(top_n) + '.txt')
    f = open(file_name, 'r', encoding='utf-8')
    qid2data = {}
    lines = f.readlines()
    for line in lines:
        line_list = line.strip().split('\t')
        qid = line_list[qid_pos]
        if(qid not in qid2data):
            temp = []
            temp.append((line_list[0:]))
            qid2data[qid] = temp
        else:
            if(len(qid2data[qid]) < top_n):
                qid2data[qid].append((line_list[0:]))
    return qid2data

# 将数据写出到文件中，针对'T'/'v'/'t'会有不同的操作
def write2file(file_name, query_answer, data_type):
    f = open(file_name, 'w', encoding='utf-8')
    if(data_type == 'T'):
        for qid in query_answer:
            num = 0
            length = len(query_answer[qid])
            for item in query_answer[qid]:
                f.write('\t'.join(item) + '\n')
                num += 1
            while(num < top_n):
                f.write('\t'.join(query_answer[qid][num % length]) + '\n')
                num += 1
    elif(data_type == 't' or data_type == 'v'):
        for qid in query_answer:
            # import pdb; pdb.set_trace()
            for item in query_answer[qid]:
                f.write('\t'.join(item) + '\n')
    f.flush()
        

if __name__ == "__main__":
    # dirName = BASE_DIR + '/runnings/train_data/webq/'
    # for top_n in [5, 10, 20, 30, 40]:
    # # for top_n in [10]:
    # # for top_n in [2, 3, 4, 6, 7, 8, 9]:
    # # for top_n in [3, 5, 7, 9, 10, 12, 14]:
    # # for top_n in [15, 20, 25, 30, 35, 40, 45, 50]:
    # # for top_n in [3, 5, 7, 9, 10, 12, 14, 15, 20, 25, 30, 35, 40, 45, 50]:
    # # for top_n in [60, 80, 100, 120]:

    #     file_name = dirName + 'bert_webq_pointwise_5244_test.txt'
    #     qid2data = get_top_n(file_name, top_n)
    #     write2file(dirName + 't_bert_top' + str(top_n) + '_from5244.txt', qid2data, 'T')

    #     file_name = dirName + 'bert_webq_pointwise_5244_dev.txt'
    #     qid2data = get_top_n(file_name, top_n)
    #     write2file(dirName + 'v_bert_top' + str(top_n) + '_from5244.txt', qid2data, 'T')

    #     # file_name = '../../rerank_data/webq/bert_webq_pointwise_5244_train.txt'
    #     # qid2data = get_top_n(file_name, top_n)
    #     # write2file('../../rerank_data/webq/T_bert_top' + str(top_n) + '_from5244.txt', qid2data, 'T')

    #     # 对交叉验证得到的训练集进行topn选取
    #     file_name = dirName + '2cv_bert_webq_pointwise_5244_train.txt'
    #     qid2data = get_top_n(file_name, top_n)
    #     write2file(dirName + 'T_cv2_bert_top' + str(top_n) + '_from5244.txt', qid2data, 'T')

    # Listwise重排序
    dirName = BASE_DIR + '/runnings/train_data/webq/'
    for top_n in [5, 10, 20]:

        file_name = dirName + '5595_rerank_listwise_negOrder_pos10_baseline_1score_webq_neg_30_42_100_test_fromrerank.txt'
        qid2data = get_top_n(file_name, top_n, qid_pos = -3)
        write2file(dirName + 't_bert_top' + str(top_n) + '_from5636.txt', qid2data, 't')

        # file_name = dirName + '5595_rerank_listwise_negOrder_pos10_baseline_1score_webq_neg_30_42_100_dev_fromrerank.txt'
        # qid2data = get_top_n(file_name, top_n)
        # write2file(dirName + 'v_bert_top' + str(top_n) + '_from5636.txt', qid2data, 'T')

        # file_name = dirName + '5530_new_bert_webq_listwise_gradual_merge_type_entity_time_ordianl_mainpath_neg_40_42_100_train_from5530.txt'
        # qid2data = get_top_n(file_name, top_n)
        # write2file(dirName + 'T_bert_top' + str(top_n) + '_from5530.txt', qid2data, 'T')
