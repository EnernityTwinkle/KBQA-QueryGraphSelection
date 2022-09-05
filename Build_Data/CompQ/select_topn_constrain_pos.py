import sys
import os
from typing import List, Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# print(BASE_DIR)
from Build_Data.common_rerank import readOrderedTrainData, selectPos, getTopN


# 将数据写出到文件中，针对'T'/'v'/'t'会有不同的操作
def write2file(file_name: str, qid2cands: Dict[str, List[List[Tuple[List[str], int]]]]) -> None:
    f = open(file_name, 'w', encoding='utf-8')
    for qid in qid2cands:
        num = 0
        # if(len(qid2cands[qid][1]) > 0):
        for item in qid2cands[qid][1]:
            # if(item[1] >= 5):
            #     import pdb; pdb.set_trace()
            f.write('\t'.join(item[0]) + '\n')
            num += 1
        for item in qid2cands[qid][0]:
            f.write('\t'.join(item[0]) + '\n')
            num += 1
    f.flush()
        

if __name__ == "__main__":
    # dirName = BASE_DIR + '/output_data/compq/'
    # for top_n in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
    #     print(top_n)
    #     ratio = 3
    #     # 对交叉验证得到的训练集进行topn选取
    #     file_name = dirName + 'bert_compq_pointwise_3835_train.txt'
    #     qid2data = readOrderedTrainData(file_name)
    #     qid2data = selectPos(qid2data, 1.0 / ratio, top_n)
    #     qid2data = getTopN(qid2data, top_n)
    #     write2file(dirName + 'compq_T_bert_constrain_pos_extra_ratio_' + str(ratio) + '_top' + str(top_n) + '_from3835.txt', qid2data)


    dirName = BASE_DIR + '/runnings/train_data/compq/'
    for top_n in [20, 30, 40, 50, 60, 80, 100]:
        print(top_n)
        ratio = 6
        # 对交叉验证得到的训练集进行topn选取
        file_name = dirName + '3711_bert_compq_pointwise_withans_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_20_42_100_train_sorted.txt'
        qid2data = readOrderedTrainData(file_name)
        qid2data = selectPos(qid2data, 1.0 / ratio, top_n)
        qid2data = getTopN(qid2data, top_n)
        write2file(dirName + 'compq_T_bert_constrain_solid20_pos_extra_ratio_' + str(ratio) + '_top' + str(top_n) + '.txt', qid2data)
        # break
