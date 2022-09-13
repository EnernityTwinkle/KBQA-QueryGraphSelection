import copy
import random

seed_sample = 100
random.seed(seed_sample)
r_shuff_sample = random.random
# neg = 1000000

# 从所有候选数据中按照一定的比例构建正例和负例训练数据,classify
def select_top_data(qid2question, que2feature, pos_only1=False, data_type = 'T', neg=1000000):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(que2feature.items(), key=lambda x: int(x[0]))
    # import pdb; pdb.set_trace()
    # for que in que2feature: # 根据查询图正例和负例信息获取相应的答案信息
    for cand_item in qid2feature_sorted:
        que = cand_item[0]
        onequery_answer = []
        features = copy.deepcopy(que2feature[que])
        # print(features)
        que_str = [item for item in set(qid2question[int(que)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, que)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, que)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, que)
                continue
            if(pos_only1): # 只取正例中的一个
                pos_features = []
                pos_features.append(random.choice(features[0]))
            else:
                pos_features = copy.deepcopy(features[0])
            for i, feature_true in enumerate(pos_features):
                onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                len_false = len(features[1])
                # actual_neg = min(neg, len_false)
                actual_neg = neg
                id_num = i * actual_neg
                while(id_num < (i + 1) * actual_neg and id_num < len_false):
                # while(id_num < (i + 1) * actual_neg):
                    # if(id_num % actual_neg == 0):
                    #     onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                    feature_false = features[1][id_num  % len_false]
                    onequery_answer.append((str(0), que_str, feature_false, que)) # 每次记录一条完整的训练数据
                    id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    return query_answer


# 从所有候选数据中按照一定的比例构建正例和负例训练数据,classify,固定正负比例
def select_classify_data_solid(qid2question, que2feature, pos_only1=False, data_type = 'T', neg=50):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(que2feature.items(), key=lambda x: int(x[0]))
    # import pdb; pdb.set_trace()
    # for que in que2feature: # 根据查询图正例和负例信息获取相应的答案信息
    for cand_item in qid2feature_sorted:
        que = cand_item[0]
        onequery_answer = []
        features = copy.deepcopy(que2feature[que])
        # print(features)
        que_str = [item for item in set(qid2question[int(que)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, que)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, que)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, que)
                continue
            if(pos_only1): # 只取正例中的一个
                pos_features = []
                pos_features.append(random.choice(features[0]))
            else:
                pos_features = copy.deepcopy(features[0])
            for i, feature_true in enumerate(pos_features):
                len_false = len(features[1])
                # actual_neg = min(neg, len_false)
                actual_neg = neg
                id_num = i * actual_neg
                onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                while(id_num < (i + 1) * actual_neg):
                # while(id_num < (i + 1) * actual_neg):
                    # if(id_num % actual_neg == 0):
                    #     onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                    feature_false = features[1][id_num  % len_false]
                    onequery_answer.append((str(0), que_str, feature_false, que)) # 每次记录一条完整的训练数据
                    id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    return query_answer


# 从所有候选数据中按照一定的比例构建正例和负例训练数据,classify
def select_pairwise_data(qid2question, que2feature, pos_only1=False, data_type = 'T', neg=50):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(que2feature.items(), key=lambda x: int(x[0]))
    # import pdb; pdb.set_trace()
    # for que in que2feature: # 根据查询图正例和负例信息获取相应的答案信息
    for cand_item in qid2feature_sorted:
        que = cand_item[0]
        onequery_answer = []
        features = copy.deepcopy(que2feature[que])
        # print(features)
        que_str = [item for item in set(qid2question[int(que)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, que)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, que)) # 每次记录一条完整的训练数据
            random.shuffle(onequery_answer, random=r_shuff_sample)
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, que)
                continue
            if(pos_only1): # 只取正例中的一个
                pos_features = []
                pos_features.append(random.choice(features[0]))
            else:
                pos_features = copy.deepcopy(features[0])
            for i, feature_true in enumerate(pos_features):
                len_false = len(features[1])
                # actual_neg = min(neg, len_false)
                actual_neg = neg
                id_num = i * actual_neg
                while(id_num < (i + 1) * actual_neg):
                # while(id_num < (i + 1) * actual_neg):
                    # if(id_num % actual_neg == 0):
                    #     onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                    feature_false = features[1][id_num  % len_false]
                    onequery_answer.append((str(1), que_str, feature_true, que)) # 每次记录一条完整的训练数据
                    onequery_answer.append((str(0), que_str, feature_false, que)) # 每次记录一条完整的训练数据
                    id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    return query_answer

# 从所有候选数据中按照正负比例选取listwise训练数据,每个正例对应的负例都是一样的，在扩大N时能保持一定的稳定性
def select_pairwise_gradual(qid2question, qid2cands, pos_only1=False, data_type='T', N=40):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(qid2cands.items(), key=lambda x: int(x[0]))
    pos_num_dic = {}
    question_num = 0
    for cand_item in qid2feature_sorted: # 根据查询图正例和负例信息获取相应的答案信息
        onequery_answer = []
        qid = cand_item[0]
        features = copy.deepcopy(qid2cands[qid])
        # print(features)
        que_str = [item for item in set(qid2question[int(qid)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, qid)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, qid)
                continue
            question_num += 1
            pos_num = len(features[0])
            # if(pos_num > N / 2):
            #     print(features[0])
            #     import pdb; pdb.set_trace()
            if(pos_num not in pos_num_dic):
                pos_num_dic[pos_num] = 1
            else:
                pos_num_dic[pos_num] += 1
            for i, feature_true in enumerate(features[0]):
            # feature_true =features[0][0]
                len_false = len(features[1])
                # actual_neg = min(neg, len_false)
                actual_neg = N
                id_num = 0
                while(id_num < actual_neg):
                # while(id_num < (i + 1) * actual_neg):
                    # if(id_num % actual_neg == 0):
                    #     onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                    feature_false = features[1][id_num  % len_false]
                    onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                    onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
                    id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    print(sorted(pos_num_dic.items(), key=lambda x: x[0]))
    print('question_num:', question_num)
    return query_answer


# 从所有候选数据中按照正负比例选取listwise训练数据,每个正例对应的负例都是一样的，在扩大N时能保持一定的稳定性
def select_pairwise_gradual_true1(qid2question, qid2cands, pos_only1=False, data_type='T', N=40):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(qid2cands.items(), key=lambda x: int(x[0]))
    pos_num_dic = {}
    question_num = 0
    for cand_item in qid2feature_sorted: # 根据查询图正例和负例信息获取相应的答案信息
        onequery_answer = []
        qid = cand_item[0]
        features = copy.deepcopy(qid2cands[qid])
        # print(features)
        que_str = [item for item in set(qid2question[int(qid)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, qid)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, qid)
                continue
            question_num += 1
            pos_num = len(features[0])
            # if(pos_num > N / 2):
            #     print(features[0])
            #     import pdb; pdb.set_trace()
            if(pos_num not in pos_num_dic):
                pos_num_dic[pos_num] = 1
            else:
                pos_num_dic[pos_num] += 1
            # for i, feature_true in enumerate(features[0]):
            feature_true =features[0][0]
            len_false = len(features[1])
            # actual_neg = min(neg, len_false)
            actual_neg = N
            id_num = 0
            while(id_num < actual_neg):
            # while(id_num < (i + 1) * actual_neg):
                # if(id_num % actual_neg == 0):
                #     onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                feature_false = features[1][id_num  % len_false]
                onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
                id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    print(sorted(pos_num_dic.items(), key=lambda x: x[0]))
    print('question_num:', question_num)
    return query_answer


# 从所有候选数据中选出n条数据，其中包含所有的正例数据，剩余个数随机选取负例数据
def select_top_n_listwise(qid2question, qid2cands, pos_only1=False, data_type='T', N=40):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(qid2cands.items(), key=lambda x: int(x[0]))
    pos_num_dic = {}
    question_num = 0
    for cand_item in qid2feature_sorted: # 根据查询图正例和负例信息获取相应的答案信息
        onequery_answer = []
        qid = cand_item[0]
        features = copy.deepcopy(qid2cands[qid])
        # print(features)
        que_str = [item for item in set(qid2question[int(qid)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, qid)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, qid)
                continue
            question_num += 1
            pos_num = len(features[0])
            # if(pos_num > N / 2):
            #     print(features[0])
            #     import pdb; pdb.set_trace()
            if(pos_num not in pos_num_dic):
                pos_num_dic[pos_num] = 1
            else:
                pos_num_dic[pos_num] += 1
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, qid)) # 每次记录一条完整的训练数据
            if(len(onequery_answer) > N / 2):
                onequery_answer = onequery_answer[0: int(N / 2)]
            actual_neg = N - len(onequery_answer)
            id_num = 0
            len_false = len(features[1])
            while(id_num < actual_neg):
                feature_false = features[1][id_num  % len_false]
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
                id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    print(sorted(pos_num_dic.items(), key=lambda x: x[0]))
    print('question_num:', question_num)
    return query_answer


# 从所有候选数据中按照正负比例选取listwise训练数据
def select_top_1_n_listwise(qid2question, qid2cands, pos_only1=False, data_type='T', N=40):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(qid2cands.items(), key=lambda x: int(x[0]))
    pos_num_dic = {}
    question_num = 0
    for cand_item in qid2feature_sorted: # 根据查询图正例和负例信息获取相应的答案信息
        onequery_answer = []
        qid = cand_item[0]
        features = copy.deepcopy(qid2cands[qid])
        # print(features)
        que_str = [item for item in set(qid2question[int(qid)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, qid)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, qid)
                continue
            question_num += 1
            pos_num = len(features[0])
            # if(pos_num > N / 2):
            #     print(features[0])
            #     import pdb; pdb.set_trace()
            if(pos_num not in pos_num_dic):
                pos_num_dic[pos_num] = 1
            else:
                pos_num_dic[pos_num] += 1
            for i, feature_true in enumerate(features[0]):
                len_false = len(features[1])
                # actual_neg = min(neg, len_false)
                actual_neg = N
                id_num = i * actual_neg
                onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                while(id_num < (i + 1) * actual_neg):
                # while(id_num < (i + 1) * actual_neg):
                    # if(id_num % actual_neg == 0):
                    #     onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                    feature_false = features[1][id_num  % len_false]
                    onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
                    id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    print(sorted(pos_num_dic.items(), key=lambda x: x[0]))
    print('question_num:', question_num)
    return query_answer

# 从所有候选数据中按照正负比例选取listwise训练数据,每个正例对应的负例都是一样的，在扩大N时能保持一定的稳定性
def select_top_1_n_listwise_gradual(qid2question, qid2cands, pos_only1=False, data_type='T', N=40):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(qid2cands.items(), key=lambda x: int(x[0]))
    pos_num_dic = {}
    question_num = 0
    for cand_item in qid2feature_sorted: # 根据查询图正例和负例信息获取相应的答案信息
        onequery_answer = []
        qid = cand_item[0]
        features = copy.deepcopy(qid2cands[qid])
        # print(features)
        que_str = [item for item in set(qid2question[int(qid)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, qid)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, qid)
                continue
            question_num += 1
            pos_num = len(features[0])
            # if(pos_num > N / 2):
            #     print(features[0])
            #     import pdb; pdb.set_trace()
            if(pos_num not in pos_num_dic):
                pos_num_dic[pos_num] = 1
            else:
                pos_num_dic[pos_num] += 1
            for i, feature_true in enumerate(features[0]):
                len_false = len(features[1])
                # actual_neg = min(neg, len_false)
                actual_neg = N
                id_num = 0
                onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                while(id_num < actual_neg):
                # while(id_num < (i + 1) * actual_neg):
                    # if(id_num % actual_neg == 0):
                    #     onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                    feature_false = features[1][id_num  % len_false]
                    onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
                    id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    print(sorted(pos_num_dic.items(), key=lambda x: x[0]))
    print('question_num:', question_num)
    return query_answer

# 从所有候选数据中按照正负比例选取listwise训练数据,每个正例对应的负例都是一样的，在扩大N时能保持一定的稳定性
def select_top_1_n_listwise_gradual_true1(qid2question, qid2cands, pos_only1=False, data_type='T', N=40):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(qid2cands.items(), key=lambda x: int(x[0]))
    pos_num_dic = {}
    question_num = 0
    for cand_item in qid2feature_sorted: # 根据查询图正例和负例信息获取相应的答案信息
        onequery_answer = []
        qid = cand_item[0]
        features = copy.deepcopy(qid2cands[qid])
        # print(features)
        que_str = [item for item in set(qid2question[int(qid)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, qid)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, qid)
                continue
            question_num += 1
            pos_num = len(features[0])
            # if(pos_num > N / 2):
            #     print(features[0])
            #     import pdb; pdb.set_trace()
            if(pos_num not in pos_num_dic):
                pos_num_dic[pos_num] = 1
            else:
                pos_num_dic[pos_num] += 1
            # for i, feature_true in enumerate(features[0]):
            feature_true =features[0][0]
            len_false = len(features[1])
            # actual_neg = min(neg, len_false)
            actual_neg = N
            id_num = 0
            onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
            while(id_num < actual_neg):
            # while(id_num < (i + 1) * actual_neg):
                # if(id_num % actual_neg == 0):
                #     onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                feature_false = features[1][id_num  % len_false]
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
                id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    print(sorted(pos_num_dic.items(), key=lambda x: x[0]))
    print('question_num:', question_num)
    return query_answer


# 从所有候选数据中按照正负比例选取listwise训练数据,同时控制每组的数量为11条，1条正例，10条负例
def select_top_1_n_listwise_11_per_group(qid2question, qid2cands, pos_only1=False, data_type='T', N=40):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(qid2cands.items(), key=lambda x: int(x[0]))
    pos_num_dic = {}
    question_num = 0
    for cand_item in qid2feature_sorted: # 根据查询图正例和负例信息获取相应的答案信息
        onequery_answer = []
        qid = cand_item[0]
        features = copy.deepcopy(qid2cands[qid])
        # print(features)
        que_str = [item for item in set(qid2question[int(qid)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, qid)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, qid)
                continue
            question_num += 1
            pos_num = len(features[0])
            # if(pos_num > N / 2):
            #     print(features[0])
            #     import pdb; pdb.set_trace()
            if(pos_num not in pos_num_dic):
                pos_num_dic[pos_num] = 1
            else:
                pos_num_dic[pos_num] += 1
            for i, feature_true in enumerate(features[0]):
                len_false = len(features[1])
                # actual_neg = min(neg, len_false)
                actual_neg = N
                id_num = i * actual_neg
                onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                per_group_num = 0
                while(id_num < (i + 1) * actual_neg):
                    # print(per_group_num)
                    if(per_group_num % 10 == 0 and per_group_num != 0):
                        onequery_answer.append((str(1), que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                    per_group_num += 1
                # while(id_num < (i + 1) * actual_neg):
                    # if(id_num % actual_neg == 0):
                    #     onequery_answer.append((str(1),que_str, feature_true, que)) # 每次记录一条完整的训练数据
                    feature_false = features[1][id_num  % len_false]
                    onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
                    id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    print(sorted(pos_num_dic.items(), key=lambda x: x[0]))
    print('question_num:', question_num)
    return query_answer


# 从所有候选数据中按照正负比例选取listwise训练数据
def select_top_1_n_listwise_pos_solid(qid2question, qid2cands, pos_only1=False, data_type='T', N=40, pos_num_save = 1, solid = False):
    query_answer = []
    num_noanswer = 0
    qid2feature_sorted = sorted(qid2cands.items(), key=lambda x: int(x[0]))
    pos_num_dic = {}
    question_num = 0
    for cand_item in qid2feature_sorted: # 根据查询图正例和负例信息获取相应的答案信息
        onequery_answer = []
        qid = cand_item[0]
        features = copy.deepcopy(qid2cands[qid])
        # print(features)
        que_str = [item for item in set(qid2question[int(qid)])][0]# 'what money is used in mozambique ?'
        # import pdb; pdb.set_trace()
        if(data_type == 't' or data_type == 'v'):
            for feature in features[0]:
                onequery_answer.append((str(1), que_str, feature, qid)) # 每次记录一条完整的训练数据
            for feature_false in features[1]:
                onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
        else:
            if(len(features[0]) == 0):
                num_noanswer += 1
                print(num_noanswer, que_str, qid)
                continue
            question_num += 1
            pos_num = len(features[0])
            # if(pos_num > N / 2):
            #     print(features[0])
            #     import pdb; pdb.set_trace()
            if(pos_num not in pos_num_dic):
                pos_num_dic[pos_num] = 1
            else:
                pos_num_dic[pos_num] += 1
            if(solid): # 正例个数固定的情况
                for pos_i in range(pos_num_save):
                    feature_true = features[0][pos_i % len(features[0])]
                    len_false = len(features[1])
                    actual_neg = N
                    id_num = pos_i * actual_neg
                    onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                    per_group_num = 0
                    while(id_num < (pos_i + 1) * actual_neg):
                        if(per_group_num % 10 == 0 and per_group_num != 0):
                            onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                        feature_false = features[1][id_num  % len_false]
                        onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
                        id_num += 1
            else:
                for i, feature_true in enumerate(features[0][0:pos_num_save]):
                    len_false = len(features[1])
                    actual_neg = N
                    id_num = i * actual_neg
                    onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                    per_group_num = 0
                    while(id_num < (i + 1) * actual_neg):
                        if(per_group_num % 10 == 0 and per_group_num != 0):
                            onequery_answer.append((str(1),que_str, feature_true, qid)) # 每次记录一条完整的训练数据
                        feature_false = features[1][id_num  % len_false]
                        onequery_answer.append((str(0), que_str, feature_false, qid)) # 每次记录一条完整的训练数据
                        id_num += 1
        if(len(onequery_answer) > 0):
            query_answer.append(onequery_answer)
    print(sorted(pos_num_dic.items(), key=lambda x: x[0]))
    print('question_num:', question_num)
    return query_answer