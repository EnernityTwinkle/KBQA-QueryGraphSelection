'''
这里用于预处理答案错误而答案类型正确的数据，防止这类数据在重排序训练时引入误差；
第一种方案：
具体而言，对于答案错误而答案类型正确的数据，将其随机替换为一个错误的类型
'''
import sys
import os
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

class AnswerInfo:
    def __init__(self, answer: str = '', answerType: str = '', label = '', qid = ''):
        self.answer = answer
        self.answerType = answerType
        self.label = label
        self.qid = qid


def readQue2Types(fileName: str):
    que2TrueType = {}
    trueType = {}
    qid2que = {}
    que2AnswerInfo = {}
    que2TrueType = {}
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        for line in lines:
            lineCut = line.strip().split('\t')
            qid = lineCut[-2]
            que = lineCut[1]
            label = lineCut[0]
            answerType = lineCut[9].split(',')
            answer = lineCut[10]
            if(qid not in qid2que):
                qid2que[qid] = que
                que2TrueType[que] = []
                que2AnswerInfo[que] = [[], []]
            if(label == '1'): # 表示是正例
                for itemType in answerType:
                    que2TrueType[que].append(itemType)
                    trueType[itemType] = 1
                que2AnswerInfo[que][1].append(AnswerInfo(answer, lineCut[9], '1', qid))
                # import pdb; pdb.set_trace()
        for i, line in enumerate(lines):
            lineCut = line.strip().split('\t')
            qid = lineCut[-1]
            que = lineCut[1]
            label = lineCut[0]
            answerType = lineCut[9].split(',')
            answer = lineCut[10]
            trueType = que2TrueType[que]
            if(label == '0'):
                flag = 0
                for itemType in answerType:
                    if(itemType in trueType):# 只有不在trueType中的才是负例，因为答案错，答案类型不一定错
                        flag = 1
                        break
                if(flag == 0):
                    que2AnswerInfo[que][0].append(AnswerInfo(answer, lineCut[9], '0', qid))
            # import pdb; pdb.set_trace()
    return que2AnswerInfo, que2TrueType


def fixAnswerErrorTypeTrue(fileName: str, que2Types: Dict[str, List[List[AnswerInfo]]], que2TrueType: Dict[str, List[str]]):
    # que2TrueType = {}
    trueType = {}
    qid2que = {}
    que2AnswerInfo = {}
    data = []
    k = 0
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        for line in lines:
            lineCut = line.strip().split('\t')
            qid = lineCut[-1]
            que = lineCut[1]
            label = lineCut[0]
            answerType = lineCut[9].split(',')
            answer = lineCut[10]
            if(qid not in qid2que):
                qid2que[qid] = que
                que2AnswerInfo[que] = [[], []]
            if(label == '1'): # 表示是正例
                # que2AnswerInfo[que][1].append(AnswerInfo(answer, lineCut[9], '1', qid))
                data.append('\t'.join(lineCut))
                # import pdb; pdb.set_trace()
            else:
                flag = 0
                trueType = que2TrueType[que]
                for itemType in answerType:
                    if(itemType in trueType):# 只有不在trueType中的才是负例，因为答案错，答案类型不一定错
                        flag = 1
                        break
                if(flag == 1):
                    length = len(que2Types[que][0])
                    # import pdb; pdb.set_trace()
                    lineCut[9] = que2Types[que][0][k % length].answerType
                    k += 1
                data.append('\t'.join(lineCut))
    return data


def readQue2AnswerInfo(fileName: str):
    que2TrueType = {}
    trueType = {}
    qid2que = {}
    que2AnswerInfo = {}
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        for line in lines:
            lineCut = line.strip().split('\t')
            qid = lineCut[-1]
            que = lineCut[1]
            label = lineCut[0]
            answerType = lineCut[9].split(',')
            answer = lineCut[10]
            if(qid not in qid2que):
                qid2que[qid] = que
                trueType = {}
                que2AnswerInfo[que] = []
            if(label == '1'):
                que2AnswerInfo[que].append(AnswerInfo(answer, lineCut[9], '1', qid))
            else:
                que2AnswerInfo[que].append(AnswerInfo(answer, lineCut[9], '0', qid))
    return que2AnswerInfo


def write2file(fileName: str, data):
    num = 0
    with open(fileName, 'w', encoding='utf-8') as fout:
        for item in data:
            fout.write(item + '\n')
    # print(len(que2AnswerInfo), num)



if __name__ == '__main__':
    fileName = BASE_DIR + '/runnings/train_data/compq/bert_compq_pairwise_4267_train_from4267.txt'
    que2AnswerInfo, que2TrueType = readQue2Types(fileName)
    data = fixAnswerErrorTypeTrue(fileName, que2AnswerInfo, que2TrueType)
    outFile = BASE_DIR + '/runnings/train_data/compq/bert_processing_compq_pairwise_4267_train_from4267.txt'
    write2file(outFile, data)

    # fileName = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_dev_all.txt'
    # que2AnswerInfo = readQue2AnswerInfoForTrain(fileName)
    # outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_dev_all_for_train.txt'
    # write2file(outFile, que2AnswerInfo)

    # fileName = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    # que2AnswerInfo = readQue2AnswerInfoForTrain(fileName)
    # outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_test_all_for_train.txt'
    # write2file(outFile, que2AnswerInfo)



    # fileName = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_dev_all.txt'
    # que2AnswerInfo = readQue2AnswerInfo(fileName)
    # outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_dev_all.txt'
    # write2file(outFile, que2AnswerInfo)

    # fileName = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    # que2AnswerInfo = readQue2AnswerInfo(fileName)
    # outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_test_all.txt'
    # write2file(outFile, que2AnswerInfo)
