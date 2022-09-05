import sys
import os
from typing import List, Dict, Tuple
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from Build_Data.WebQ.build_answer_type_train_data import AnswerInfo

def readQue2AnswerInfo(fileName: str):
    que2AnswerInfo = {}
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        for line in lines:
            lineCut = line.strip().split('\t')
            label = lineCut[0]
            que = lineCut[1]
            answerType = lineCut[2]
            answer = lineCut[3].lower()
            qid = lineCut[-1]
            if(que not in que2AnswerInfo):
                que2AnswerInfo[que] = [[], []]
            que2AnswerInfo[que][int(label)].append(AnswerInfo(answer, answerType, label, qid))
    return que2AnswerInfo


def sampleData(que2AnswerInfo: Dict[str, List[List[AnswerInfo]]], negNum: int = 4):
    que2data = {}
    for que in que2AnswerInfo:
        answerInfos = que2AnswerInfo[que]
        data = []
        for posAnswer in answerInfos[1]:
            data.append(posAnswer)
            length = len(answerInfos[0])
            if(length == 0):
                print(que)
                data = []
                break
            for i in range(negNum):
                data.append(answerInfos[0][i % length])
        que2data[que] = data
    return que2data

            
def write2file(fileName: str, que2data):
    with open(fileName, 'w', encoding='utf-8') as fout:
        for que in que2data:
            for answerInfo in que2data[que]:
                fout.write(answerInfo.label + '\t' + que + '\t' + answerInfo.answerType \
                    + '\t' + answerInfo.answer + '\t' + answerInfo.qid + '\n')



if __name__ == "__main__":
    que2AnswerInfo = readQue2AnswerInfo(BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_train_all_for_train.txt')
    for i in [4, 9, 19]:
        que2data = sampleData(que2AnswerInfo, i)
        outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_train_1_' + str(i) + '.txt'
        write2file(outFile, que2data)