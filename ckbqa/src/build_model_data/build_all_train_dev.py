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
    fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
    questions = getQuestions(fileName)
    random.shuffle(questions)
    devQuestions = questions[0: int(len(questions) * 0.2)]
    devQuestionsDic = {item: 1 for item in devQuestions}
    # fileFeature = 'trainset_seq_9750_pos_f106_neg03_0308'
    # outFileFeature = 'ccks_9750_0308'
    fileFeature = '9750_trainset_withAnswerType_seq'
    outFileFeature = 'ccks_9750_0308'
    # candsFile = BASE_DIR + '/data/candidates/seq/trainset_seq_' + fileFeature + '.txt'
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
    devNeg = 1000000
    for que in que2posCands:
        if(que in devQuestionsDic):
            temp = []
            for cand in que2posCands[que]:
                temp.append((que, cand))
            for cand in que2negCands[que][0: devNeg]:
                temp.append((que, cand))
            devData.append(temp)
    devFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/dev_20%_neg' + str(devNeg) + '.txt'
    write2file(trainData=devData, fileName=devFile)

    # negNums = [5, 10, 15, 20]
    negNums = [1000000]
    for negNum in negNums:
        trainData = []
        outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/train_neg' + str(negNum) + '.txt'
        for index, que in enumerate(que2posCands):
            if(que in devQuestionsDic):
                continue
            if(index % 100 == 0):
                print(negNum, index)
            temp = []
            for cand in que2posCands[que]:
                temp.append((que, cand))
            for cand in que2negCands[que][0: negNum]:
                temp.append((que, cand))
            trainData.append(temp)
        print('train group num:', len(trainData))
        write2file(trainData=trainData, fileName=outFile)



if __name__ == "__main__":
    buildTrainDataNoCopyPos()