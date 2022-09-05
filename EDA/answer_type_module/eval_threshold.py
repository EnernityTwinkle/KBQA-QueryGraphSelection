import json
import sys
import os
from typing import List, Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# 读取预测的得分文件
def read_prediction_scores(file_name: str) -> List[float]:
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    scores: List[float] = []
    for i, line in enumerate(lines):
        scores.append(float(line.strip()))
    return scores


# 读取候选文件
def read_cands(file_name: str, scores: List[float]) -> Dict[str, List[Tuple[float, str]]]:
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    qid2cands: Dict[str, List[Tuple[float, str]]] = {}# 记录每个问句对应的候选信息
    current_qid = ''
    assert len(scores) == len(lines)
    for i, line in enumerate(lines):
        line_cut = line.strip().split('\t')
        current_qid = line_cut[-1]
        # if(line_cut[0] == '1' and scores[i] < 0.1):
        #     print(scores[i], line.strip())
        if(current_qid not in qid2cands):
            qid2cands[current_qid] = []
            qid2cands[current_qid].append((scores[i], line.strip()))
            # import pdb; pdb.set_trace()
        else:
            qid2cands[current_qid].append((scores[i], line.strip()))
    return qid2cands


# 判断正确答案类型位次排第几
def judgeTypeRank(qid2cands: Dict[str, List[Tuple[float, str]]]):
    for qid in qid2cands:
        cands = qid2cands[qid]
        cands.sort(key=lambda x:x[0], reverse=True)
        length = len(cands)
        for i, item in enumerate(cands):
            if(item[1][0] == '1' and i > length * 0.6):
                print(i, i * 1.0 / length, item)
        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    scoreFile = BASE_DIR + '/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_9_42_50/prediction_test_all'
    candsFile = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    scores = read_prediction_scores(scoreFile)
    qid2cands = read_cands(candsFile, scores)
    judgeTypeRank(qid2cands)
    