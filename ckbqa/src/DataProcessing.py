import sys
import os
import json
from typing import List, Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class DataProcessing:
    def __init__(self) -> None:
        self.trainFile = BASE_DIR + '/data/NLPCC/nlpcc_2016_kbqa_train_linking'
        self.testFile = BASE_DIR + '/data/NLPCC/nlpcc_2016_kbqa_test_linking'
        self.trainFileCCKS2019 = BASE_DIR + '/data/CCKS2019/task6ckbqa_train_2019.txt'
        self.qid2que: Dict[str, str] = {}
        self.qid2answer: Dict[str, str] = {}
        self.qid2queCCKS2019: Dict[str, str] = {}
        self.qid2answerCCKS2019: Dict[str, str] = {}
        self.readTrainData()
        self.readTrainDataCCKS2019()

    def readTrainData(self):
        with open(self.trainFile, 'r', encoding = 'utf-8') as fread:
            lines = fread.readlines()
            length = len(lines)
            for i in range(0, length, 4):
                lineCut = lines[i].strip().split('\t')
                idBegin = lineCut[0].index('id=')
                idValue = lineCut[0][idBegin + 3: -1]
                self.qid2que[idValue] = lineCut[1]
                answer = lines[i + 2].split('\t')[1][0:-1]
                self.qid2answer[idValue] = answer
                # import pdb; pdb.set_trace()

    def readTrainDataCCKS2019(self):
        with open(self.trainFileCCKS2019, 'r', encoding='utf-8') as fread:
            lines = fread.readlines()
            length = len(lines)
            for i in range(0, length, 4):
                index = lines[i].index(':')
                qid = lines[i][1:index]
                que = lines[i][index + 1:-1]
                self.qid2queCCKS2019[qid] = que
                answer = lines[i + 2].rstrip()
                self.qid2answerCCKS2019[qid] = answer
                # import pdb; pdb.set_trace()



    def readTestData(self):
        with open(self.testFile, 'r', encoding = 'utf-8') as fread:
            lines = fread.readlines()
            length = len(lines)
            for i in range(0, length, 4):
                lineCut = lines[i].strip().split('\t')
                idBegin = lineCut[0].index('id=')
                idValue = lineCut[0][idBegin + 3: -1]
                self.qid2que[idValue] = lineCut[1]
                answer = lines[i + 2].strip().split('\t')[1]
                self.qid2answer[idValue] = answer



if __name__ == "__main__":
    dataProcessing = DataProcessing()
    dataProcessing.readTrainData()
