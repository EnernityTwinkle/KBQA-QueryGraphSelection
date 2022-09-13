import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from Build_Data.read_query_graph import *


if __name__ == "__main__":
    # *************************按照正负例构建listwise数据，并且标记出每种子路径的位置***************
    init_dir_name = BASE_DIR + '/runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    entity_dic = read_entity(init_dir_name)
    qid2comp_dic = read_comp(init_dir_name)
    qid2cands = read_query_graph(init_dir_name, entity_dic, qid2comp_dic)
    qid2cands = get_pos_neg_accord_f1(qid2cands)
    f = open(BASE_DIR + '/data/webq_qid2question.pkl', 'rb')
    qid2question = pickle.load(f)
    qid2cands_train, qid2cands_dev, qid2cands_test = split_data_webq(qid2cands)
    for N in [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140]:
    # for N in [5]:
        file_name = BASE_DIR + '/runnings/train_data/webq/webq_rank1_f01_gradual_label_position_1_' + str(N) + '_type_entity_time_ordinal_mainpath_'
        train_data = select_top_1_n_listwise_gradual(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=N)
        write2file_label_position(file_name + 'is_train.txt', train_data)

    # 验证集和测试集都是使用全集
    dev_data = select_top_1_n_listwise(qid2question, qid2cands_dev, pos_only1=False, data_type='v', N=N)
    write2file_label_position(file_name + 'dev_all.txt', dev_data)
    test_data = select_top_1_n_listwise(qid2question, qid2cands_test, pos_only1=False, data_type='t', N=N)
    write2file_label_position(file_name + 'test_all.txt', test_data)
    