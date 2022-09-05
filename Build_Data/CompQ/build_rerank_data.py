'''
构建重排序数据的两个约束：topn没有正例时，需要加一个正例；正例个数不超过n/2
'''

import sys
import os
from typing import List, Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# print(BASE_DIR)
from Build_Data.common_rerank import constrainTrainData, readOrderedTrainData, selectPos, getTopNPrediction


# 将数据写出到文件中，针对'T'/'v'/'t'会有不同的操作
def write2file(file_name: str, qid2cands: Dict[str, List[List[Tuple[List[str], int]]]]) -> None:
    f = open(file_name, 'w', encoding='utf-8')
    for qid in qid2cands:
        num = 0
        for cands in qid2cands[qid]:
            for item in cands:
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
    # for top_n in [50, 60, 70, 80, 90, 100, 110, 120]:
    # for top_n in [5, 10, 20, 30, 40]:
    for top_n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        print(top_n)
        # 对交叉验证得到的训练集进行topn选取
        # file_name = dirName + 'compq_train_2cv.txt'
        # qid2data = getTopNPrediction(file_name, top_n) # 得到topn数据
        # qid2dataFull = readOrderedTrainData(file_name)
        # qid2data = constrainTrainData(qid2data, qid2dataFull, top_n)
        # write2file(dirName + 'compq_T_bert_2cv_constrain_top' + str(top_n) + '.txt', qid2data)
        # # break

        # devFile = dirName + 'bert_compq_pointwise_3835_dev.txt'
        # qid2data = getTopNPrediction(devFile, top_n)
        # write2file(dirName + 'compq_v_top' + str(top_n) + '.txt', qid2data)

        testFile = dirName + 'bert_compq_pointwise_3835_test.txt'
        qid2data = getTopNPrediction(testFile, top_n)
        write2file(dirName + 'compq_t_top' + str(top_n) + '.txt', qid2data)
