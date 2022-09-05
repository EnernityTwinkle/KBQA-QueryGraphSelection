'''
分析查询图排序阶段top-n的性能
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


# 计算最高宏观F1值
def calMaxF1CompQ(qid2data: Dict[str, List[List[Tuple[List[str], int]]]]) -> float:
    f1Sum = 0.0
    num = 800
    for qid in qid2data:
        cands = qid2data[qid]
        tempF1 = 0.0
        for cand in cands[0]:
            # import pdb; pdb.set_trace()
            if(float(cand[0][-3]) > tempF1):
                tempF1 = float(cand[0][-3])

        for cand in cands[1]:
            # import pdb; pdb.set_trace()
            if(float(cand[0][-3]) > tempF1):
                tempF1 = float(cand[0][-3])
        f1Sum += tempF1
    return f1Sum / num


def calMaxF1WebQ(qid2data: Dict[str, List[List[Tuple[List[str], int]]]]) -> float:
    f1Sum = 0.0
    num = 2032
    # print(len(qid2data))
    for qid in qid2data:
        cands = qid2data[qid]
        tempF1 = 0.0
        for cand in cands[0]:
            # import pdb; pdb.set_trace()
            if(float(cand[0][-3]) > tempF1):
                tempF1 = float(cand[0][-3])
        for cand in cands[1]:
            # import pdb; pdb.set_trace()
            if(float(cand[0][-3]) > tempF1):
                tempF1 = float(cand[0][-3])
        f1Sum += tempF1
    # print(f1Sum)
    return f1Sum / num

        

if __name__ == "__main__":
    dirName = BASE_DIR + '/runnings/train_data/compq/'
    top_n = 1
    while(top_n <= 40):
        if(top_n == 1 or top_n % 1 == 0):
            testFile = dirName + '4454_new_bert_compq_listwise_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_120_42_100_test_sorted_4440.txt'
            qid2data = getTopNPrediction(testFile, top_n)
            f1 = calMaxF1CompQ(qid2data)
            print(top_n, f1)
            # import pdb; pdb.set_trace()
        top_n += 1


    # dirName = BASE_DIR + '/runnings/train_data/webq/'
    # top_n = 1
    # while(top_n <= 40):
    #     if(top_n == 1 or top_n % 1 == 0):
    #         # testFile = dirName + 'bert_webq_pointwise_5244_test.txt'
    #         testFile = '/home/chenwenliang/jiayonghui/gitlab/rerankinglab/runnings/train_data/webq/5530_new_bert_webq_listwise_gradual_merge_type_entity_time_ordianl_mainpath_neg_40_42_100_test_from5530.txt'
    #         qid2data = getTopNPrediction(testFile, top_n)
    #         f1 = calMaxF1WebQ(qid2data)
    #         print(top_n, f1)
    #         # import pdb; pdb.set_trace()
    #     top_n += 1
