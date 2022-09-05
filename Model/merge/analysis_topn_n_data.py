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
        if(current_qid not in qid2cands):
            qid2cands[current_qid] = []
            qid2cands[current_qid].append((scores[i], line.strip()))
            # import pdb; pdb.set_trace()
        else:
            qid2cands[current_qid].append((scores[i], line.strip()))
    return qid2cands

# 对每个问句对应的候选按照得分进行排序
def sort_cands(qid2cands: Dict[str, List[Tuple[float, str]]]) -> Dict[str, List[Tuple[float, str]]]:
    for qid in qid2cands:
        cands = qid2cands[qid]
        cands.sort(key=lambda x:x[0], reverse=True)
    return qid2cands

# 选取出top1错误但top2正确的候选数据
def selectTop1FalseTop2True(qid2cands: Dict[str, List[Tuple[float, str]]]) -> None:
    for qid in qid2cands:
        cands = qid2cands[qid]
        if(len(cands) > 2):
            if(cands[0][1][0] == '0' and cands[1][1][0] == '1'):
                print(cands[0])
                print(cands[1])

if __name__ == "__main__":
    scoreFile = BASE_DIR + '/runnings/model/webq/2bert_answer_type_bert_webq_pointwise_2linear_neg_100_42_50/prediction_test'
    candsFile = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    scores = read_prediction_scores(scoreFile)
    qid2cands = read_cands(candsFile, scores)
    qid2cands = sort_cands(qid2cands)
    selectTop1FalseTop2True(qid2cands)