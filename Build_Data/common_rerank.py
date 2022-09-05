
from typing import List, Dict, Tuple
import math
import random
import copy
random.seed(100)

def getTopN(qid2cands: Dict[str, List[List[Tuple[List[str], int]]]], n: int) \
                -> Dict[str, List[List[Tuple[List[str], int]]]]:
    qid2candsNew = copy.deepcopy(qid2cands)
    for qid in qid2candsNew:
        threshold = math.ceil(n / 2.0)
        qid2candsNew[qid][1] = qid2candsNew[qid][1][0: threshold]
        qid2candsNew[qid][0] = qid2candsNew[qid][0][0: n - len(qid2candsNew[qid][1])]
        length = len(qid2candsNew[qid][0])
        num = len(qid2candsNew[qid][0]) + len(qid2candsNew[qid][1])
        while(num < n):
            qid2candsNew[qid][0].append(qid2candsNew[qid][0][num % length])
            num += 1
    return qid2candsNew

def readOrderedTrainData(file_name: str) -> Dict[str, List[List[Tuple[List[str], int]]]]:
    fread = open(file_name, 'r', encoding='utf-8')
    qid2cands: Dict[str, List[List[Tuple[List[str], int]]]] = {}
    i = 0
    for line in fread:
        lineSplit = line.strip().split('\t')
        lineSplit[-6] = ' '.join(lineSplit[-6].split(' ')[0:512])
        # import pdb; pdb.set_trace()
        qid = lineSplit[-2]
        label = int(lineSplit[0])
        if(qid not in qid2cands):
            i = 0
            qid2cands[qid] = [[], []]
            qid2cands[qid][label].append((lineSplit[0: ], i))
        else:
            i += 1
            qid2cands[qid][label].append((lineSplit[0: ], i))
        # import pdb; pdb.set_trace()
    return qid2cands


def selectPos(qid2cands: Dict[str, List[List[Tuple[List[str], int]]]], PosRatio: float, top_n: int) \
                -> Dict[str, List[List[Tuple[List[str], int]]]]:
    qid2candsNew = copy.deepcopy(qid2cands)
    for qid in qid2candsNew:
        length = len(qid2candsNew[qid][0]) + len(qid2candsNew[qid][1])
        threshold = math.ceil(length * PosRatio)
        top_n = 20 # 强制前20
        for i, item in enumerate(qid2candsNew[qid][1]):
            if(item[1] > threshold or item[1] >= top_n):
                qid2candsNew[qid][1] = qid2candsNew[qid][1][0: i]
                break
    return qid2candsNew


def getTopNTrain(qid2cands: Dict[str, List[List[Tuple[List[str], int]]]], n: int) \
                -> Dict[str, List[List[Tuple[List[str], int]]]]:
    qid2candsNew = copy.deepcopy(qid2cands)
    for qid in qid2candsNew:
        threshold = math.ceil(n / 2.0)
        qid2candsNew[qid][1] = qid2candsNew[qid][1][0: threshold]
        qid2candsNew[qid][0] = qid2candsNew[qid][0][0: n - len(qid2candsNew[qid][1])]
        length = len(qid2candsNew[qid][0])
        num = len(qid2candsNew[qid][0]) + len(qid2candsNew[qid][1])
        while(num < n):
            qid2candsNew[qid][0].append(qid2candsNew[qid][0][num % length])
            num += 1
    return qid2candsNew

# 从已经拍好序的候选文件中选出top_n构建重排序数据
def getTopNPrediction(file_name: str, top_n: int) -> Dict[str, List[List[Tuple[List[str], int]]]]:
    f = open(file_name, 'r', encoding='utf-8')
    qid2data: Dict[str, List[List[Tuple[List[str], int]]]] = {}
    lines = f.readlines()
    for line in lines:
        line_list = line.strip().split('\t')
        qid = line_list[-2]
        label = int(line_list[0])
        if(qid not in qid2data):
            i = 0
            qid2data[qid] = [[], []]
            qid2data[qid][label].append((line_list[0: ], i))
        else:
            i += 1
            if(i < top_n):
                qid2data[qid][label].append((line_list[0: ], i))
    return qid2data


def constrainTrainData(qid2data: Dict[str, List[List[Tuple[List[str], int]]]], 
                        qid2candsFull: Dict[str, List[List[Tuple[List[str], int]]]], topN: int) \
                            -> Dict[str, List[List[Tuple[List[str], int]]]]:
    for qid in qid2data:
        cands = qid2data[qid]
        trueNum = len(cands[1])
        length = len(cands[0]) + len(cands[1])
        if(trueNum == 0 and len(qid2candsFull[qid][1]) > 0):
            cands[1].append(qid2candsFull[qid][1][0])
            cands[0] = cands[0][0: -1]
        if(trueNum * 2 > length):
            deleteNum = trueNum - length // 2
            cands[1] = cands[1][0:-deleteNum]
            for item in qid2candsFull[qid][0]:
                if(item[1] >= length):
                    cands[0].append(item)
        newLength = len(cands[0]) + len(cands[1])
        if(newLength > topN):
            moreNum = newLength - topN
            cands[0] = cands[0][0: -moreNum]
        else:
            lessNum = topN - newLength
            negNum = len(cands[0])
            for i in range(lessNum):
                cands[0].append(cands[0][i % negNum])
            # import pdb; pdb.set_trace()
        if(len(cands[0]) + len(cands[1]) != topN):
            import pdb; pdb.set_trace()
    return qid2data



        


def convertPosToRandomPos(qid2cands: Dict[str, List[List[Tuple[List[str], int]]]],\
                            qid2candsAll: Dict[str, List[List[Tuple[List[str], int]]]]) \
                                -> Dict[str, List[List[Tuple[List[str], int]]]]:
    qid2candsNew = copy.deepcopy(qid2cands)
    for qid in qid2candsAll:
        candsAll = qid2candsAll[qid]
        random.shuffle(candsAll[1])
        posNum = len(qid2candsNew[qid][1])
        qid2candsNew[qid][1] = candsAll[1][0:posNum]
    return qid2candsNew


def convertNegToRandomNeg(qid2cands: Dict[str, List[List[Tuple[List[str], int]]]],\
                            qid2candsAll: Dict[str, List[List[Tuple[List[str], int]]]]) \
                                -> Dict[str, List[List[Tuple[List[str], int]]]]:
    qid2candsNew = copy.deepcopy(qid2cands)
    for qid in qid2candsAll:
        candsAll = qid2candsAll[qid]
        random.shuffle(candsAll[0])
        negNum = len(qid2candsNew[qid][0])
        qid2candsNew[qid][0] = candsAll[0][0:negNum]
    return qid2candsNew
    

def randomCands(qid2cands: Dict[str, List[List[Tuple[List[str], int]]]]) \
                -> Dict[str, List[List[Tuple[List[str], int]]]]:
    qid2candsNew = copy.deepcopy(qid2cands)
    for qid in qid2candsNew:
        random.shuffle(qid2candsNew[qid][0])
        random.shuffle(qid2candsNew[qid][1])
    return qid2candsNew


def randomPosCands(qid2cands: Dict[str, List[List[Tuple[List[str], int]]]]) \
                -> Dict[str, List[List[Tuple[List[str], int]]]]:
    qid2candsNew = copy.deepcopy(qid2cands)
    for qid in qid2candsNew:
        random.shuffle(qid2candsNew[qid][1])
    return qid2candsNew


def randomNegCands(qid2cands: Dict[str, List[List[Tuple[List[str], int]]]]) \
                -> Dict[str, List[List[Tuple[List[str], int]]]]:
    qid2candsNew = copy.deepcopy(qid2cands)
    for qid in qid2candsNew:
        random.shuffle(qid2candsNew[qid][0])
    return qid2candsNew