import sys
import os
from typing import List, Dict, Tuple
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from src.rerank.build_data.select_topn_constrain_pos import readOrderedTrainData

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
    return qid2dataNew


def selectTopBasedNegLow(qid2data, N):
    qid2dataNew = {}
    for qid in qid2data:
        qid2dataNew[qid] = []
        for candPos in qid2data[qid][1]:
            if(len(qid2data[qid][0]) > 0):
                qid2dataNew[qid].append(candPos)
                negCands = qid2data[qid][0]
                negNum = len(negCands)
                for i in range(N):
                    qid2dataNew[qid].append(negCands[-1 - (i % negNum)])
    return qid2dataNew


def write2file(fileName, qid2data):
    f = open(fileName, 'w', encoding='utf-8')
    for qid in qid2data:
        for cand in qid2data[qid]:
            f.write(str(cand[0]) + '\n')
            # import pdb; pdb.set_trace()
    f.flush()



if __name__ == "__main__":
    # train用这个
    ########## CCKS2021-CompKBQA
    dirName = BASE_DIR + '/data/train_data/ccks_comp/'
    for top_n in [10, 20, 30, 40, 50]:
        print(top_n)
        ratio = 4
        # 对交叉验证得到的训练集进行topn选取
        file_name = dirName + '7490_ccks_comp_listwise_bert_devneg100_seq150_CE_group_20_1_42_100_train_from.txt'
        qid2data = readOrderedTrainData(file_name)
        qid2data = selectTopBasedNegHigh(qid2data, top_n - 1)
        write2file(dirName + 'rerank_T_bert_negOrder_1_n' + '_top' + str(top_n) + '.txt', qid2data)
