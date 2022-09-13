import sys
import os
import random
import json
random.seed(100)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_processing import readCandsInfo, getQuestions

MAX_LENGTH = 150

def write2file(trainData, fileName):
    with open(fileName, 'w', encoding='utf-8') as fout:
        for trainGroup in trainData:
            for item in trainGroup:
                fout.write(str((item[0], json.dumps(item[1], ensure_ascii=False))) + '\n')


def buildTrainDataNoCopyPos():
    # fileName = BASE_DIR + '/data/data_from_pzhou/trainset/train_query_triple.txt'
    # questions = getQuestions(fileName)
    # devQuestionsDic = {}
    # fileFeature = '9734_nopad_multiTypes_0115'
    # outFileFeature = 'multi_types_0116_nopad'
    # fileFeature = 'trainset_seq_9535_multiTypes'
    # outFileFeature = 'ccks_comp'
    fileFeature = 'stagg_trainset_seq_8696_multiTypes'
    outFileFeature = 'stagg_ccks_comp'
    candsFile = BASE_DIR + '/data/candidates/seq/' + fileFeature + '.txt'
    que2cands = readCandsInfo(candsFile)
    que2posCands = {}
    que2negCands = {}
    for que in que2cands:
        
        if(que not in que2posCands):
            que2posCands[que] = []
            que2negCands[que] = []
        cands = que2cands[que]
        for cand in cands:
            if(cand["label"] == 1):
                que2posCands[que].append(cand)
            else:
                que2negCands[que].append(cand)
        random.shuffle(que2posCands[que])
        random.shuffle(que2negCands[que])
    # negNums = [5, 10, 15, 20]
    negNums = [120, 100, 80, 60, 40, 20, 15, 10, 5]
    for negNum in negNums:
        trainData = []
        outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/train_neg' + str(negNum) + '.txt'
        for index, que in enumerate(que2posCands):
            if(index % 100 == 0):
                print(negNum, index)
            for posCand in que2posCands[que]:
                lenNegCands = len(que2negCands[que])
                if(lenNegCands == 0):
                    print('0:', que)
                    break
                temp = []
                temp.append((que, posCand))
                for i in range(negNum):
                    temp.append((que, que2negCands[que][i % lenNegCands]))
                trainData.append(temp)
        print('train group num:', len(trainData))
        write2file(trainData=trainData, fileName=outFile)

def buildDevDataNoCopyPos():
    # fileName = BASE_DIR + '/data/data_from_pzhou/devset/dev_query_triple.txt'
    # devQuestions = getQuestions(fileName)
    # devQuestionsDic = {item: 1 for item in devQuestions}
    # fileFeature = '9708_nopad_multiTypes_0115'
    # outFileFeature = 'multi_types_0116_nopad'
    # fileFeature = 'devset_seq_9444_multiTypes'
    # outFileFeature = 'ccks_comp'
    fileFeature = 'stagg_devset_seq_8408_multiTypes'
    outFileFeature = 'stagg_ccks_comp'
    candsFile = BASE_DIR + '/data/candidates/seq/' + fileFeature + '.txt'
    que2cands = readCandsInfo(candsFile)
    que2posCands = {}
    que2negCands = {}
    for que in que2cands:
        if(que not in que2posCands):
            que2posCands[que] = []
            que2negCands[que] = []
        cands = que2cands[que]
        for cand in cands:
            if(cand["label"] == 1):
                que2posCands[que].append(cand)
            else:
                que2negCands[que].append(cand)
        random.shuffle(que2posCands[que])
        random.shuffle(que2negCands[que])
    devData = []
    devNeg = 100
    for que in que2posCands:
        temp = []
        for cand in que2posCands[que]:
            temp.append((que, cand))
        for cand in que2negCands[que][0: devNeg]:
            temp.append((que, cand))
        devData.append(temp)
    devFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/dev_20%_neg' + str(devNeg) + '.txt'
    write2file(trainData=devData, fileName=devFile)


if __name__ == "__main__":
    buildTrainDataNoCopyPos()
    buildDevDataNoCopyPos()