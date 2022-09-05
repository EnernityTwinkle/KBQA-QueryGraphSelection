import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

class AnswerInfo:
    def __init__(self, answer: str = '', answerType: str = '', label = '', qid = ''):
        self.answer = answer
        self.answerType = answerType
        self.label = label
        self.qid = qid


def readQue2AnswerInfoForTrain(fileName: str):
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
            if(label == '1'): # 表示是正例
                for itemType in answerType:
                    trueType[itemType] = 1
                que2AnswerInfo[que].append(AnswerInfo(answer, lineCut[9], '1', qid))
                # import pdb; pdb.set_trace()
            else:
                flag = 0
                for itemType in answerType:
                    if(itemType in trueType):# 只有不在trueType中的才是负例，因为答案错，答案类型不一定错
                        flag = 1
                        break
                if(flag == 0):
                    que2AnswerInfo[que].append(AnswerInfo(answer, lineCut[9], '0', qid))
                    # import pdb; pdb.set_trace()
    return que2AnswerInfo


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


def write2file(fileName: str, que2AnswerInfo):
    num = 0
    with open(fileName, 'w', encoding='utf-8') as fout:
        for que in que2AnswerInfo:
            answerInfos = que2AnswerInfo[que]
            if(len(answerInfos) > 2):
                num += 1
            else:
                print(que)
            for answerInfo in answerInfos:
                fout.write(answerInfo.label + '\t' + que + '\t' + answerInfo.answerType \
                    + '\t' + answerInfo.answer + '\t' + answerInfo.qid + '\n')
    print(len(que2AnswerInfo), num)



if __name__ == '__main__':
    # fileName = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_train_all.txt'
    # que2AnswerInfo = readQue2AnswerInfoForTrain(fileName)
    # outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_train_all_for_train.txt'
    # write2file(outFile, que2AnswerInfo)

    # fileName = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_dev_all.txt'
    # que2AnswerInfo = readQue2AnswerInfoForTrain(fileName)
    # outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_dev_all_for_train.txt'
    # write2file(outFile, que2AnswerInfo)

    # fileName = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    # que2AnswerInfo = readQue2AnswerInfoForTrain(fileName)
    # outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_test_all_for_train.txt'
    # write2file(outFile, que2AnswerInfo)



    fileName = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_dev_all.txt'
    que2AnswerInfo = readQue2AnswerInfo(fileName)
    outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_dev_all.txt'
    write2file(outFile, que2AnswerInfo)

    fileName = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    que2AnswerInfo = readQue2AnswerInfo(fileName)
    outFile = BASE_DIR + '/runnings/train_data/webq/webq_only_answer_info_test_all.txt'
    write2file(outFile, que2AnswerInfo)
