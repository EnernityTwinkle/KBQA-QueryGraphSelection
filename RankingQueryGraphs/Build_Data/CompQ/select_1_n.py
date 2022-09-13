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
        for item in qid2cands[qid]:
            # import pdb; pdb.set_trace()
            # if(item[1] >= 5):
            #     import pdb; pdb.set_trace()
            f.write('\t'.join(item[0]) + '\n')
            num += 1
    f.flush()
        

def selectTopBasedNegHigh(qid2data, N):
    qid2dataNew = {}
    for qid in qid2data:
        qid2dataNew[qid] = []
        for candPos in qid2data[qid][1]:
            if(len(qid2data[qid][0]) > 0):
                qid2dataNew[qid].append(candPos)
                negCands = qid2data[qid][0]
                negNum = len(negCands)
                for i in range(N):
                    qid2dataNew[qid].append(negCands[i % negNum])
        # import pdb; pdb.set_trace()
    return qid2dataNew


def selectTopBasedNegHighPosRatio(qid2data, N, position, ratio = 0.25):
    qid2dataNew = {}
    for qid in qid2data:
        qid2dataNew[qid] = []
        # position = 20
        for candPos in qid2data[qid][1]:
            if(candPos[1] > position):
                continue
            # import pdb; pdb.set_trace()
            if(len(qid2data[qid][0]) > 0):
                qid2dataNew[qid].append(candPos)
                negCands = qid2data[qid][0]
                negNum = len(negCands)
                for i in range(N):
                    qid2dataNew[qid].append(negCands[i % negNum])
        # import pdb; pdb.set_trace()
    return qid2dataNew


if __name__ == "__main__":


    # dirName = BASE_DIR + '/runnings/train_data/compq/'
    # for top_n in [20, 30, 40, 50, 60, 80, 100]:
    #     print(top_n)
    #     ratio = 6
    #     # 对交叉验证得到的训练集进行topn选取
    #     file_name = dirName + '3711_bert_compq_pointwise_withans_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_20_42_100_train_sorted.txt'
    #     qid2data = readOrderedTrainData(file_name)
    #     qid2data = selectTopBasedNegHigh(qid2data, top_n - 1)
    #     write2file(dirName + 'compq_T_bert_negOrder_1_n_top' + str(top_n) + '.txt', qid2data)
    #     # break

    dirName = BASE_DIR + '/runnings/train_data/compq/'
    # for top_n in [20, 30, 40, 50, 60, 80, 100]:
    for top_n in [5, 10]:
        print(top_n)
        ratio = 6
        # 对交叉验证得到的训练集进行topn选取
        file_name = dirName + '3711_bert_compq_pointwise_withans_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_20_42_100_train_sorted.txt'
        qid2data = readOrderedTrainData(file_name)
        qid2data = selectTopBasedNegHighPosRatio(qid2data, top_n - 1, 10)
        write2file(dirName + 'compq_T_bert_negOrder_pos10_1_n_top' + str(top_n) + '.txt', qid2data)
        # break