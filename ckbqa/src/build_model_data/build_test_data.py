import sys
import os
import random
import json
random.seed(100)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_processing import readCandsInfo

def write2file(trainData, fileName):
    with open(fileName, 'w', encoding='utf-8') as fout:
        for trainGroup in trainData:
            for item in trainGroup:
                fout.write(str((item[0], json.dumps(item[1], ensure_ascii=False))) + '\n')            

if __name__ == "__main__":
    # fileFeature = '2_1031'
    # fileFeature = '8680_nopad_multiTypes_0115'
    # outFileFeature = 'multi_types_0116_nopad'
    fileFeature = '8677'
    outFileFeature = 'ccks_9750_0308'
    # fileFeature = '7488_64_multiTypes_0114'
    # outFileFeature = 'multi_types'
    candsFile = BASE_DIR + '/data/candidates/seq/testset_seq_' + fileFeature + '.txt'
    #############################STAGG#########################
    # fileFeature = 'stagg_testset_seq_8479_0308'
    # outFileFeature = 'stagg_ccks'
    # # candsFile = BASE_DIR + '/data/candidates/seq/trainset_seq_' + fileFeature + '.txt'
    # candsFile = BASE_DIR + '/data/candidates/seq/' + fileFeature + '.txt'
    ######## 针对STAGG方法 #######################
    # fileFeature = '1115'
    # outFileFeature = 'stagg_1115'
    # candsFile = BASE_DIR + '/data/candidates/seq/stagg_8263_testset_seq_' + fileFeature + '.txt'
    ######################
    que2cands = readCandsInfo(candsFile)
    que2posCands = {}
    que2negCands = {}
    trainData = []
    for que in que2cands:
        if(que not in que2posCands):
            que2posCands[que] = []
            que2negCands[que] = []
        cands = que2cands[que]
        temp = []
        for cand in cands:
            temp.append((que, cand))
        trainData.append(temp)
    outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/testset_model_data.txt'
    write2file(trainData=trainData, fileName=outFile)
    # import pdb; pdb.set_trace()