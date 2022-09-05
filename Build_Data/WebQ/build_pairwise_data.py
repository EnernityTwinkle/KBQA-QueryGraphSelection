import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
x = []

from Build_Data.read_query_graph import *


if __name__ == "__main__":
    # init_dir_name = '../../runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    # entity_dic = read_entity(init_dir_name)
    # qid2comp_dic = read_comp(init_dir_name)
    # qid2cands = read_query_graph(init_dir_name, entity_dic, qid2comp_dic)
    # qid2cands = get_pos_neg_accord_f1(qid2cands)
    # f = open('/data2/yhjia/kbqa_sp/webq_qid2question.pkl', 'rb')
    # qid2question = pickle.load(f)
    # qid2cands_train, qid2cands_dev, qid2cands_test = split_data_webq(qid2cands)
    # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
    # # file_name = './webq_pairwise_rank1_p01_f01_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
    # # file_name = '/data2/yhjia/kbqa_train_data/WebQ/pairwise/webq_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'

    # # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
    # # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_mainpath_'
    # # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_mainrel_'
    # train_data = select_pairwise_data(qid2question, qid2cands_train, pos_only1=False, data_type='T', neg=NEG_NUM)
    # write2file(file_name + '_train_all.txt', train_data)
    # # 验证集和测试集都是使用全集
    # dev_data = select_pairwise_data(qid2question, qid2cands_dev, pos_only1=False, data_type='v', neg=NEG_NUM)
    # write2file(file_name + 'dev_all.txt', dev_data)
    # test_data = select_pairwise_data(qid2question, qid2cands_test, pos_only1=False, data_type='t', neg=NEG_NUM)
    # write2file(file_name + 'test_all.txt', test_data)

    # *****************************标记出不同子路径的位置*******************
    init_dir_name = '../../runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    entity_dic = read_entity(init_dir_name)
    qid2comp_dic = read_comp(init_dir_name)
    qid2cands = read_query_graph(init_dir_name, entity_dic, qid2comp_dic)
    qid2cands = get_pos_neg_accord_f1(qid2cands)
    f = open('../../data/webq_qid2question.pkl', 'rb')
    qid2question = pickle.load(f)
    qid2cands_train, qid2cands_dev, qid2cands_test = split_data_webq(qid2cands)
    for NEG_NUM in [5, 10, 20, 40, 60, 80, 100, 120, 140]:
        # file_name = './webq_one_answer_label_position_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        # file_name = './webq_600rank1_f01_label_position_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        # file_name = './webq_pairwise_rank1_p01_f01_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        # file_name = '/data2/yhjia/kbqa_train_data/WebQ/pairwise/webq_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        file_name = BASE_DIR + '/runnings/train_data/webq/' + 'webq_rank1_f01_label_position_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        train_data = select_pairwise_gradual(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=NEG_NUM)
        write2file_label_position(file_name + '_train.txt', train_data)
    # 验证集和测试集都是使用全集
    file_name = BASE_DIR + '/runnings/train_data/webq/' + 'pairwise_'
    dev_data = select_pairwise_data(qid2question, qid2cands_dev, pos_only1=False, data_type='v')
    write2file_label_position(file_name + 'dev_all.txt', dev_data)
    test_data = select_pairwise_data(qid2question, qid2cands_test, pos_only1=False, data_type='t')
    write2file_label_position(file_name + 'test_all.txt', test_data)