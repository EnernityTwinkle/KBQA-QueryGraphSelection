import sys
import os
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Model.cal_f1 import cal_f1_with_position

def readScores(fileName: str) -> List[float]:
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        scores: List[float] = []
        for line in lines:
            scores.append(float(line.strip()))
        return scores


def weight_score(scores1: List[float], scores2: List[float], weight: float) -> List[float]:
    scores: List[float] = []
    for i, item in enumerate(scores1):
        score = weight * item + (1.0 - weight) * scores2[i]
        # if(scores2[i] > 0.95):
        #     score = item + 0.1
        # if(scores2[i] < threshold): # 如果答案类型相似度低于某个阈值，那么该候选得分降低
        #     score = item / 2.0
            # score = 0.0
            # print(scores2[i])
        scores.append(score)
    return scores


def write2file(fileName: str, scores: List[float]) -> None:
    with open(fileName, 'w', encoding='utf-8') as fout:
        for score in scores:
            fout.write(str(score) + '\n')
        fout.flush()

def test(threshold = 0.0):
    fileNameDev = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    fileNameSim = BASE_DIR + '/runnings/model/webq/2bert_answer_type_bert_webq_pointwise_2linear_neg_100_42_50/prediction_test'
    fileNameAnswer = BASE_DIR + '/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_4_42_50/prediction_test_all'
    scoresSim = readScores(fileNameSim)
    scoresAnswer = readScores(fileNameAnswer)
    # for weight in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
    p_max = 0
    r_max = 0
    f_max = 0
    weight_max = 0
    threshold_max = 0
    for weight in [1.0]:
        # for threshold in [0, 0.0001]:
        # threshold = 0.006
        # threshold = 0.0
        scores = weight_score(scoresSim, scoresAnswer, weight, threshold)
        write2file('./scores.txt', scores)
        p, r, f = cal_f1_with_position('./scores.txt', fileNameDev, 't', -2)

if __name__ == "__main__":
    # test(0.006)
    # import pdb; pdb.set_trace()
    fileNameDev = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_dev_all.txt'
    fileNameSim = BASE_DIR + '/runnings/model/webq/2bert_answer_type_bert_webq_pointwise_2linear_neg_100_42_50/prediction_valid'
    fileNameAnswer = BASE_DIR + '/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_9_42_50/prediction_dev_all'
    # fileNameDev = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    # fileNameSim = BASE_DIR + '/runnings/model/webq/2bert_answer_type_bert_webq_pointwise_2linear_neg_100_42_50/prediction_test'
    # fileNameAnswer = BASE_DIR + '/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_9_42_50/prediction_test_all'
    scoresSim = readScores(fileNameSim)
    scoresAnswer = readScores(fileNameAnswer)
    
    p_max = 0
    r_max = 0
    f_max = 0
    weight_max = 0
    threshold_max = 0
    for weight in [1.0, 0.99, 0.98, 0.97, 0.96]:
        print(weight)
        scores = weight_score(scoresSim, scoresAnswer, weight)
        write2file('./scores.txt', scores)
        p, r, f = cal_f1_with_position('./scores.txt', fileNameDev, 'v', -2)
        if(f > f_max):
            print(p, r, f, weight)
            p_max = p
            r_max = r
            f_max = f
            weight_max = weight
    print(p_max, r_max, f_max, weight_max, threshold_max)

    