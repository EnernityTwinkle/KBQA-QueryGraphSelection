'''
答案类型对，答案不一定对；答案类型错，答案一定错。
'''
import sys
import os
from typing import List, Dict, Tuple
import copy
from collections import OrderedDict


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


def constrain_score(scores1: List[float], scores2: List[float], threshold: float) -> List[float]:
    assert len(scores1) == len(scores2)
    scores: List[float] = []
    for i, item in enumerate(scores1):
        # score = weight * item + (1.0 - weight) * scores2[i]
        # if(scores2[i] > 0.95):
        #     score = item + 0.1
        score = item
        if(scores2[i] < threshold): # 如果答案类型相似度低于某个阈值，那么降低该候选得分
            score -= 0.9
        scores.append(score)
    return scores

# 读取候选文件
def read_cands(file_name: str, scoresType: List[float], scoresSim: List[float]) -> Dict[str, List[Tuple[float, float, str]]]:
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    qid2cands: Dict[str, List[Tuple[float, float, str]]] = OrderedDict()# 记录每个问句对应的候选信息
    current_qid = ''
    assert len(scoresType) == len(lines) and len(scoresType) == len(scoresSim)
    for i, line in enumerate(lines):
        line_cut = line.strip().split('\t')
        current_qid = line_cut[-1]
        # if(line_cut[0] == '1' and scores[i] < 0.1):
        #     print(scores[i], line.strip())
        if(current_qid not in qid2cands):
            qid2cands[current_qid] = []
            qid2cands[current_qid].append((scoresType[i], scoresSim[i], line.strip()))
            # import pdb; pdb.set_trace()
        else:
            qid2cands[current_qid].append((scoresType[i], scoresSim[i], line.strip()))
    return qid2cands

# 按照答案类型所在位次进行得分约束
def constrainScoreAccordingOrder(qid2cands: Dict[str, List[Tuple[float, float, str]]], order: float) -> List[float]:
    qid2candsSort = copy.deepcopy(qid2cands)
    scores: List[float] = []
    for qid in qid2cands:
        candsSort = qid2candsSort[qid]
        candsSort.sort(key = lambda x: x[0], reverse=True)
        length = len(candsSort) * order
        for i, item in enumerate(qid2cands[qid]):
            indexOrder = candsSort.index(item)
            if(indexOrder > length):
                scores.append(item[1] - 0.9)
                if(item[2][0] == '1'):
                    print(item)
            else:
                scores.append(item[1])
            # import pdb; pdb.set_trace()
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
    print(threshold)
    scores = constrain_score(scoresSim, scoresAnswer, threshold)
    write2file('./scores.txt', scores)
    p, r, f = cal_f1_with_position('./scores.txt', fileNameDev, 't', -2)


def testConstrainOrder(threshold: float = 0.0):
    fileNameDev = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    fileNameSim = BASE_DIR + '/runnings/model/webq/2bert_answer_type_bert_webq_pointwise_2linear_neg_100_42_50/prediction_test'
    fileNameAnswer = BASE_DIR + '/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_4_42_50/prediction_test_all'
    scoresSim = readScores(fileNameSim)
    scoresAnswer = readScores(fileNameAnswer)
    qid2cands = read_cands(fileNameDev, scoresAnswer, scoresSim)
    scores = constrainScoreAccordingOrder(qid2cands, threshold)
    print(threshold)
    # scores = constrain_score(scoresSim, scoresAnswer, threshold)
    write2file('./scores.txt', scores)
    p, r, f = cal_f1_with_position('./scores.txt', fileNameDev, 't', -2)

if __name__ == "__main__":
    # test(0.1)
    for i in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        testConstrainOrder(i)
        import pdb; pdb.set_trace()
    fileNameDev = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_dev_all.txt'
    fileNameSim = BASE_DIR + '/runnings/model/webq/2bert_answer_type_bert_webq_pointwise_2linear_neg_100_42_50/prediction_valid'
    # fileNameSim = BASE_DIR + '/runnings/model/webq/bert_webq_pointwise_gradual_merge_type_entity_time_ordianl_mainpath_neg_50_42_100/prediction_valid'
    fileNameAnswer = BASE_DIR + '/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_4_42_50/prediction_dev_all'
    # fileNameDev = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    # fileNameSim = BASE_DIR + '/runnings/model/webq/2bert_answer_type_bert_webq_pointwise_2linear_neg_100_42_50/prediction_test'
    # fileNameAnswer = BASE_DIR + '/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_9_42_50/prediction_test_all'
    scoresSim = readScores(fileNameSim)
    scoresAnswer = readScores(fileNameAnswer)
    qid2cands = read_cands(fileNameDev, scoresAnswer, scoresSim)
    p_max = 0
    r_max = 0
    f_max = 0
    weight_max = 0
    threshold_max = 0
    threshold = 0.0
    ###################用答案类型得分进行约束##########################
    # while(threshold < 0.1):
    #     print(threshold)
    #     # scores = constrain_score(scoresSim, scoresAnswer, threshold)
    #     scores = constrainScoreAccordingOrder(qid2cands)
    #     write2file('./scores.txt', scores)
    #     p, r, f = cal_f1_with_position('./scores.txt', fileNameDev, 'v', -2)
    #     if(f > f_max):
    #         print(p, r, f, threshold)
    #         p_max = p
    #         r_max = r
    #         f_max = f
    #         threshold_max = threshold
    #     threshold += 0.002
    # print(p_max, r_max, f_max, threshold_max)
    ##################用答案类型位次进行约束####################
    threshold = 1.0
    while(threshold > 0):
        print(threshold)
        # scores = constrain_score(scoresSim, scoresAnswer, threshold)
        scores = constrainScoreAccordingOrder(qid2cands, threshold)
        write2file('./scores.txt', scores)
        p, r, f = cal_f1_with_position('./scores.txt', fileNameDev, 'v', -2)
        if(f > f_max):
            print(p, r, f, threshold)
            p_max = p
            r_max = r
            f_max = f
            threshold_max = threshold
        threshold -= 0.1
    print(p_max, r_max, f_max, threshold_max)

    