import sys
import os
import json
import copy
import re
from typing import List, Dict, Tuple
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_processing import getQuestionsAndTypes, getQuestionsWithComplex, readQuestionType
from src.eda.error_analysis import read_prediction_scores, read_cands, get_qid2maxf1, get_qid2f1


if __name__ == "__main__":
    # fileName = BASE_DIR + '/dataset/ccks2021/multi_types_questions_dataset.json'
    # questions, que2type = getQuestionsAndTypes(fileName)
    # fileName = BASE_DIR + '/src/question_classification/build_data/que_newType_test.txt'
    fileName = BASE_DIR + '/dataset/ccks2021/que_newType_test.txt'
    questions = getQuestionsWithComplex(fileName)
    que2type = readQuestionType(fileName)
    # file_name = '/data2/yhjia/kbqa/ckbqa/src/model_train/models/7491_8590_listwise_nosigmoid_bert_devneg50_seq150_CE_group_20_1_42_100/prediction_var_noentity_only'
    file_name = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/7372_pointwise_ccks_comp_bert_trainexamples_devneg50_seq150_CE_neg15_24_42_1/prediction'
    scores = read_prediction_scores(file_name)
    # file_name = BASE_DIR + '/data/train_data/testset_model_data.txt'
    # file_name = BASE_DIR + '/data/train_data/3_1101/testset_model_data.txt'
    file_name = BASE_DIR + '/data/train_data/ccks_comp/testset_model_data.txt' # 测试集查询图候选文件
    # file_name = BASE_DIR + '/data/train_data/y_6_4_1102/dev_20%_neg50.txt'
    qid2cands, qid2pos, qid_list = read_cands(file_name, -2)
    qid2maxf1, qid2maxcand = get_qid2maxf1(qid2cands, qid2pos, qid_list)
    qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, scores, qid2maxcand)
    quetype2true = {}
    quetype2all = {}
    for que in questions:
        queType = que2type[que]
        if(queType not in quetype2true):
            quetype2true[queType] = 0
            quetype2all[queType] = 0
        if(que in qid2f1):
            quetype2true[queType] += qid2f1[que]
        quetype2all[queType] += 1
    for queType in quetype2true:
        print(queType, quetype2true[queType] * 1.0 / quetype2all[queType])
    print(quetype2all)
    # import pdb; pdb.set_trace()
    # pass