import os
import sys
import json
import random
import copy
import pickle
from typing import List
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from Build_Data.sample_pos_and_neg import select_top_data, select_pairwise_data, select_top_n_listwise, select_classify_data_solid, select_top_1_n_listwise,\
                                select_top_1_n_listwise_pos_solid, select_top_1_n_listwise_11_per_group, \
                                select_top_1_n_listwise_gradual, select_top_1_n_listwise_gradual_true1, \
                                select_pairwise_gradual_true1, select_pairwise_gradual
seed = 100
random.seed(seed)
r_shuff = random.random


class QueryGraphForTrain:
    def __init__(self, main_path = [], entity_path = [], time_path = [],
                     type_path = [], ordinal_path = [], p = 0, r = 0, f1 = 0,\
                          entityId: List[str] = [], relationId: List[str] = [],\
                              answerType: str = '', answerStr:str = ''):
        '''
        这里包含QueryGraph中的每条边信息，其中entity_path指包含所有实体信息的路径；time_path指描述时间约束的路径；type_sparql指描述类型约束的路径；ordinal_path指序数词约束的路径；
        而entity_sparql、time_sparql、type_sparql和ordinal_sparql分别对应每种路径的sparql语句。
        注：描述成list结构是因为每种路径可能会包含不同层级的三元组，这里初步设计为可以分离的模式
        '''
        self.main_path = main_path

        self.entity_path = entity_path
        
        self.time_path = time_path
      
        self.type_path = type_path
     
        self.ordinal_path = ordinal_path
    
        self.p = p
        self.r = r
        self.f1 = f1
        self.entityId = entityId
        self.relationId = relationId
        self.answerType = answerType
        self.answerStr = answerStr

    def set_entity_path(self, entity_path):
        self.entity_path = entity_path
    def set_entity_sparql(self, entity_sparql):
        self.entity_sparql = entity_sparql
    def set_time_path(self, time_path):
        self.time_path = time_path
    def set_time_sparql(self, time_sparql):
        self.time_sparql = time_sparql
    def set_type_path(self, type_path):
        self.type_path = type_path
    def set_type_sparql(self, type_sparql):
        self.type_sparql = type_sparql
    def set_ordinal_path(self, ordinal_path):
        self.ordinal_path = ordinal_path
    def set_ordianl_sparql(self, ordinal_sparql):
        self.ordinal_sparql = ordinal_sparql
    def set_answer(self, answer):
        self.answer = answer
    def serialize(self):
        print(self.main_path, self.entity_path, self.time_path, self.type_path, self.ordinal_path, self.p, self.r, self.f1)


def normalize_entity(entity):
    entity = entity.replace('_', ' ').lower()
    return entity

def normalize_relation(rel):
    rel = rel.split('.')[-1].replace('_', ' ').lower()
    return rel

def normalize_answer(ans):
    ans = ans.replace('\t', ', ').lower()
    return ans


# 读取实体表信息
def read_entity(init_dir_name):
    entity_dic = {}
    for root, dirs, files in os.walk(init_dir_name):
        # print(root, dirs, files)
        for dir_name in dirs: # 针对每组问句对应的文件夹进行处理
            file_names = os.listdir(root + dir_name)
            for file_name in file_names:
                if('_links' in file_name):
                    # import pdb; pdb.set_trace()
                    f = open(init_dir_name + dir_name + '/' + file_name, 'r', encoding = 'utf-8')
                    for line in f:
                        line_json = json.loads(line.strip())
                        if(len(line_json) != 9):
                            print('长度不等于9')
                            import pdb; pdb.set_trace()
                        if(line_json[6][0] != 'value'):
                            print('value 所在位置不一致')
                            import pdb; pdb.set_trace()
                        value = line_json[6][1]
                        if(line_json[7][0] != 'name'):
                            print('name 所在位置不一致')
                            import pdb; pdb.set_trace()
                        name = line_json[7][1]
                        if(value not in entity_dic):
                            entity_dic[value] = name
    return entity_dic

# 读取比较信息，一般只有时间约束和序数词约束会有意义
def read_comp(init_dir_name):
    qid2comp_dic = {}
    for root, dirs, files in os.walk(init_dir_name):
        # print(root, dirs, files)
        for dir_name in dirs: # 针对每组问句对应的文件夹进行处理
            file_names = os.listdir(root + dir_name)
            for file_name in file_names:
                if('_links' in file_name):
                    # import pdb; pdb.set_trace()
                    qid = file_name[0:4]
                    qid2comp_dic[qid] = []
                    f = open(init_dir_name + dir_name + '/' + file_name, 'r', encoding = 'utf-8')
                    for i, line in enumerate(f):
                        line_json = json.loads(line.strip())
                        if(len(line_json) != 9):
                            print('长度不等于9')
                            import pdb; pdb.set_trace()
                        if(line_json[5][0] != 'comp'):
                            print('comp 所在位置不一致')
                            import pdb; pdb.set_trace()
                        comp = line_json[5][1]
                        qid2comp_dic[qid].append(comp)
    return qid2comp_dic



# 将freebase中的关系转为常规词,num为取几个层级之间的词，层级之间用'.'分隔
def trans_relid2words(rel_id, num = 1):
    words_list = rel_id.split('.')[0-num:]
    words_str = ' '.join(words_list)
    words_str = words_str.replace('_', ' ')
    return words_str


# 得到一条候选的基本信息
def get_info_of_cand(line, entity_dic, comp_list):
    main_path = []
    entity_path = []
    time_path = []
    type_path = []
    ordinal_path = []
    line_json = json.loads(line.strip())
    p = line_json['p']
    r = line_json['r']
    f1 = line_json['f1']
    ans_str = ' '.join(line_json['ans_str'].replace('\t', ', ').split(' ')[0:100])# 保留所有答案
    # ans_str = ''
    # ans_str = ','.join(line_json['ans_str'].split('\t')[0:3]) # 保留一个答案
    answerType = line_json['answer_type'].split('\t')
    answerTypeWords = ''
    for item in answerType:
        if(len(answerTypeWords) != 0):
            answerTypeWords += ', ' + trans_relid2words(item)
        else:
            answerTypeWords += trans_relid2words(item)
    raw_paths = line_json['raw_paths']
    entityId = []
    relationId = []
    # import pdb; pdb.set_trace()
    for subpath in raw_paths:
        if(len(subpath) != 4):
            print(subpath)
            import pdb; pdb.set_trace()
        if(subpath[0] == 'Main'):
            entity_str = entity_dic[subpath[2]].replace('_', ' ')
            entityId.append(subpath[2])
            for i, item in enumerate(subpath[3]):
                subpath[3][i] = trans_relid2words(item)
                relationId.append(item)
            core_rel_str = ' -- '.join(subpath[3])
            # core_rel_str = subpath[3][-1]
            main_path.append(entity_str)
            main_path.append(core_rel_str)
            main_path.append(ans_str)
        elif(subpath[0] == 'Entity'):
            entity_str = entity_dic[subpath[2]].replace('_', ' ')
            # if(len(subpath[3]) > 1): # 存在大于1的情况：[['Main', 4, 'm.06s5sq', ['release date s', 'release date']], 
            #     #['Entity', 0, 'm.03rjj', ['!film.film_regional_release_date.release_date', 'film.film_regional_release_date.film_release_region']]]
            #     print('实体约束关系大于1')
            #     import pdb; pdb.set_trace()
            for i, item in enumerate(subpath[3]):
                subpath[3][i] = trans_relid2words(item)
                relationId.append(item)
            # constrain_rel = ' -- '.join(subpath[3])
            constrain_rel = subpath[3][-1]
            entity_path.append(constrain_rel)
            entity_path.append(entity_str)
            # import pdb; pdb.set_trace()
            # if(len(subpath[3]) == 2):
            #     print(subpath)
            #     import pdb; pdb.set_trace()
        elif(subpath[0] == 'Time'):
            time_str = subpath[2]
            for i, item in enumerate(subpath[3]):
                subpath[3][i] = trans_relid2words(item)
                relationId.append(item)
                # subpath[3][i] = trans_relid2words(item, 2)
            # time_rel = ' -- '.join(subpath[3])
            time_rel = subpath[3][-1]
            time_path.append(time_rel)
            time_path.append(time_str)
            # print(subpath)
            # import pdb; pdb.set_trace()
        elif(subpath[0] == 'Type'):
            type_str = trans_relid2words(subpath[2])
            type_path.append(type_str)
        elif(subpath[0] == 'Ordinal'):
            ordinal_str = subpath[2]
            for i, item in enumerate(subpath[3]):
                subpath[3][i] = trans_relid2words(item, 1)
                relationId.append(item)
            # ordinal_rel = ' -- '.join(subpath[3])
            ordinal_rel = subpath[3][-1]
            ordinal_path.append(ordinal_rel)
            ordinal_path.append(comp_list[subpath[1]])
            ordinal_path.append(ordinal_str)
            # print(ordinal_path)
            # import pdb; pdb.set_trace()
        else:
            print(subpath)
            import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    return main_path, entity_path, time_path, type_path, ordinal_path, p, r, f1, entityId, relationId, answerTypeWords, ans_str

# 读取查询图信息
def read_query_graph(init_dir_name, entity_dic, qid2comp_dic):
    num = 0
    que2cands_dic = {}
    for root, dirs, files in os.walk(init_dir_name):
        # print(root, dirs, files)
        for dir_name in dirs: # 针对每组问句对应的文件夹进行处理
            file_names = os.listdir(root + dir_name)
            for file_name in file_names:
                qid = file_name[0:4]
                comp_list = qid2comp_dic[qid]
                # if(int(qid) % 100 != 0):
                #     continue
                if('_schema' in file_name):
                    f = open(init_dir_name + '/' + dir_name + '/' + file_name, 'r', encoding = 'utf-8')
                    for line in f:
                        if(qid not in que2cands_dic):
                            que2cands_dic[qid] = []
                        # print(qid)
                        # import pdb; pdb.set_trace()
                        main_path, entity_path, time_path, type_path, ordinal_path, p, r, f1,\
                             entityId, relationId, answerTypeWords, ans_str = get_info_of_cand(line, entity_dic, comp_list)
                        query_graph = QueryGraphForTrain(main_path = main_path, entity_path = entity_path, time_path = time_path,\
                                                        type_path = type_path, ordinal_path = ordinal_path, p = p, r = r, f1 = f1,\
                                                            entityId=entityId, relationId=relationId, answerType = answerTypeWords,\
                                                                answerStr = ans_str)
                        # query_graph.serialize()
                        que2cands_dic[qid].append(query_graph)
                        num += 1
                        # que2cands_dic.append((p, r, f, ans_str, ))
                        # import pdb; pdb.set_trace()
    print("候选总个数:{%d}, 过滤后的问句总个数:{%d}, 平均每个问句对应的候选个数:{%f}" % \
        (num, len(que2cands_dic), num * 1.0 / len(que2cands_dic)))
    return que2cands_dic

def takeF1(item):
    return float(item.f1)

# 将候选分为正例和负例
def get_pos_neg_accord_f1(que2feature):
    qid2cand = {}
    for qid in que2feature:
        features = copy.deepcopy(que2feature[qid])
        # print(features)
        features.sort(key=takeF1, reverse=True) # 对一个问句对应的候选查询图进行排序
        # features = que2feature[qid]
        max_f1 = float(features[0].f1)
        # import pdb; pdb.set_trace()
        for feature in features:
            if(qid not in qid2cand):
                qid2cand[qid] = [[], []]
                if(float(feature.f1) == max_f1 and max_f1 > 0.1):
                # if(feature.f1 > 0.1):
                # if(float(feature.f1) == max_f1 and float(feature.f1) > 0.1 and float(feature.p) > 0.1):
                # if(float(feature.f1) > 0.9 and float(feature.p) > 0.1):
                    qid2cand[qid][0].append(feature)# 0存放正确的候选
                # elif(float(feature.f1) == 0):
                #     qid2cand[qid][1].append(feature)# 1存放错误的候选
                else:
                    qid2cand[qid][1].append(feature)# 1存放错误的候选
            else:
                if(float(feature.f1) == max_f1 and max_f1 > 0.1):
                # if(float(feature.f1) == max_f1 and float(feature.f1) > 0.1 and float(feature.p) > 0.1):
                # if(feature.f1 > 0.1):
                # if(float(feature.f1) > 0.9 and float(feature.p) > 0.1):
                    qid2cand[qid][0].append(feature)# 0存放正确的候选
                # elif(float(feature.f1) == 0):
                #     qid2cand[qid][1].append(feature)# 1存放错误的候选
                else:
                    qid2cand[qid][1].append(feature)# 1存放错误的候选
        random.shuffle(qid2cand[qid][1], random=r_shuff)# 对负例进行随机化，排序baseline生成top的顺序影响
        random.shuffle(qid2cand[qid][0], random=r_shuff)
    return qid2cand

# 将数据分为训练集，验证集和测试集
def split_data_compq(qid2cands):
    qid2cands_train = {}
    qid2cands_dev = {}
    qid2cands_test = {}
    for qid in qid2cands:
        qid_int = int(qid)
        if(qid_int < 1000):
            qid2cands_train[qid] = qid2cands[qid]
        elif(qid_int >= 1000 and qid_int < 1300):
            qid2cands_dev[qid] = qid2cands[qid]
        else:
            qid2cands_test[qid] = qid2cands[qid]
    return qid2cands_train, qid2cands_dev, qid2cands_test


# 将数据分为训练集，验证集和测试集
def split_data_webq(qid2cands):
    qid2cands_train = {}
    qid2cands_dev = {}
    qid2cands_test = {}
    for qid in qid2cands:
        qid_int = int(qid)
        if(qid_int < 3023):
            qid2cands_train[qid] = qid2cands[qid]
        elif(qid_int >= 3023 and qid_int < 3778):
            qid2cands_dev[qid] = qid2cands[qid]
        else:
            qid2cands_test[qid] = qid2cands[qid]
    return qid2cands_train, qid2cands_dev, qid2cands_test


# 将数据写出到文件
def write2file(file_name, query_answer):
    f = open(file_name, 'w', encoding='utf-8')
    num = 0
    for onequery_item in query_answer:
        num += 1
        for item in onequery_item:
            # import pdb; pdb.set_trace()
            query_graph = item[2]
            cand_str = ''
            # *********************约束路径放在主路径之前****************************
            if(query_graph.type_path != []):
                cand_str += ' '.join(query_graph.type_path) + '. '
                # 重复主路径，answer用类型代替
                # cand_str += ' '.join(query_graph.main_path[0:2]) + ' ' + query_graph.type_path[0] + '. '
            # cand_str += '\t'
            if(query_graph.entity_path != []):
                cand_str += ' '.join(query_graph.entity_path) + '. '
                # cand_str += query_graph.entity_path[0] + ' is ' + query_graph.entity_path[1] + '. '
            # cand_str += '\t'
            if(query_graph.time_path != []):
                cand_str += ' '.join(query_graph.time_path) + '. '
                # cand_str += query_graph.time_path[0] + ' is ' + query_graph.time_path[1] + '. '
                # cand_str += 'time is ' + query_graph.time_path[1] + '. '
                # print(item[1].lower())
                # print('time:', cand_str)
            # cand_str += '\t'
            if(query_graph.ordinal_path != []):
                # cand_str += ' ' + ' '.join(query_graph.ordinal_path) + '.'
                # cand_str += query_graph.ordinal_path[0] + ' is ' + query_graph.ordinal_path[1] + '. '
                cand_str += query_graph.ordinal_path[0] + ' ' + query_graph.ordinal_path[1] + ' ' + query_graph.ordinal_path[2] + '. '
                # cand_str += 'rank is ' + query_graph.ordinal_path[1] + '. '
                # print('ordinal:', cand_str)
                # cand_str += ' ' + query_graph.ordinal_path[0] + ' rank ' + query_graph.ordinal_path[1] + '.'
            # cand_str += '\t'
            # ********************************************************************
            cand_str += ' '.join(query_graph.main_path[0:3]) + '.' # 包含主实体、关系和答案
            # cand_str += ' '.join(query_graph.main_path[1]) + '.' # 包含主关系
            # cand_str += query_graph.main_path[0] +  ' <' + query_graph.main_path[1] + '>.'
            # cand_str += query_graph.main_path[0] +  ' <' + query_graph.main_path[1] + '> ' + query_graph.main_path[2] + '.'
            # cand_str = query_graph.main_path[0] + ' ' + query_graph.main_path[1] + ' is ' + query_graph.main_path[2] + '.' # 包含主实体、关系和答案
            # if(query_graph.type_path != []):
            #     cand_str += ' '.join(query_graph.type_path) + '. '
            #     # 重复主路径，answer用类型代替
            #     # cand_str += ' '.join(query_graph.main_path[0:2]) + ' ' + query_graph.type_path[0] + '. '
            # if(query_graph.entity_path != []):
            #     cand_str += ' '.join(query_graph.entity_path) + '. '
            #     # cand_str += query_graph.entity_path[0] + ' is ' + query_graph.entity_path[1] + '. '
            # if(query_graph.time_path != []):
            #     cand_str += ' '.join(query_graph.time_path) + '. '
            #     # cand_str += query_graph.time_path[0] + ' is ' + query_graph.time_path[1] + '. '
            #     # cand_str += 'time is ' + query_graph.time_path[1] + '. '
            #     # print(item[1].lower())
            #     # print('time:', cand_str)
            # if(query_graph.ordinal_path != []):
            #     # cand_str += ' ' + ' '.join(query_graph.ordinal_path) + '.'
            #     # cand_str += query_graph.ordinal_path[0] + ' is ' + query_graph.ordinal_path[1] + '. '
            #     cand_str += query_graph.ordinal_path[0] + ' ' + query_graph.ordinal_path[1] + ' ' + query_graph.ordinal_path[2] + '. '
            f.write(item[0] + '\t' + item[1].lower() + '\t' + cand_str.lower() + '\t' +\
                    str(query_graph.p) + '\t' + str(query_graph.r) + '\t' + str(query_graph.f1) + '\t' + item[3] + '\n')
    f.flush()
    print('问句个数：', num)

# 将数据写出到文件
def write2file_label_position(file_name, query_answer):
    f = open(file_name, 'w', encoding='utf-8')
    num = 0
    for onequery_item in query_answer:
        num += 1
        for item in onequery_item:
            # import pdb; pdb.set_trace()
            query_graph = item[2]
            cand_str = ''
            # *********************约束路径放在主路径之前****************************
            if(query_graph.type_path != []):
                cand_str += ' '.join(query_graph.type_path) + '. '
                # 重复主路径，answer用类型代替
                # cand_str += ' '.join(query_graph.main_path[0:2]) + ' ' + query_graph.type_path[0] + '. '
            cand_str += '\t'
            if(query_graph.entity_path != []):
                cand_str += ' '.join(query_graph.entity_path) + '. '
                # cand_str += query_graph.entity_path[0] + ' is ' + query_graph.entity_path[1] + '. '
            cand_str += '\t'
            if(query_graph.time_path != []):
                cand_str += ' '.join(query_graph.time_path) + '. '
                # cand_str += query_graph.time_path[0] + ' is ' + query_graph.time_path[1] + '. '
                # cand_str += 'time is ' + query_graph.time_path[1] + '. '
                # print(item[1].lower())
                # print('time:', cand_str)
            cand_str += '\t'
            if(query_graph.ordinal_path != []):
                # cand_str += ' ' + ' '.join(query_graph.ordinal_path) + '.'
                # cand_str += query_graph.ordinal_path[0] + ' is ' + query_graph.ordinal_path[1] + '. '
                cand_str += query_graph.ordinal_path[0] + ' ' + query_graph.ordinal_path[1] + ' ' + query_graph.ordinal_path[2] + '. '
                # cand_str += 'rank is ' + query_graph.ordinal_path[1] + '. '
                # print('ordinal:', cand_str)
                # cand_str += ' ' + query_graph.ordinal_path[0] + ' rank ' + query_graph.ordinal_path[1] + '.'
            cand_str += '\t'
            # ********************************************************************
            cand_str += ' '.join(query_graph.main_path[0:3]) + '.' # 包含主实体、关系和答案
            # import pdb; pdb.set_trace()
            f.write(item[0] + '\t' + item[1].lower() + '\t' + cand_str.lower() + \
                        '\t' + '##'.join(query_graph.entityId) +\
                        '\t' + '##'.join(query_graph.relationId) + '\t' + \
                        query_graph.answerType  + '\t' + query_graph.answerStr + '\t' + \
                        str(query_graph.p) + '\t' + str(query_graph.r) + '\t' + str(query_graph.f1) \
                         + '\t' + item[3] + '\n')
    f.flush()
    print('问句个数：', num)


# 将数据写出到文件,加额外的词使查询图序列更像一句话
def write2file_label_position_type(file_name, query_answer):
    f = open(file_name, 'w', encoding='utf-8')
    num = 0
    for onequery_item in query_answer:
        num += 1
        for item in onequery_item:
            # import pdb; pdb.set_trace()
            query_graph = item[2]
            cand_str = ''
            # *********************约束路径放在主路径之前****************************
            if(query_graph.type_path != []):
                cand_str += 'type is ' + ' '.join(query_graph.type_path) + '. '
                # 重复主路径，answer用类型代替
                # cand_str += ' '.join(query_graph.main_path[0:2]) + ' ' + query_graph.type_path[0] + '. '
            cand_str += '\t'
            if(query_graph.entity_path != []):
                # cand_str += ' '.join(query_graph.entity_path) + '. '
                cand_str += query_graph.entity_path[0] + ' is ' + query_graph.entity_path[1] + '. '
            cand_str += '\t'
            if(query_graph.time_path != []):
                # cand_str += ' '.join(query_graph.time_path) + '. '
                cand_str += query_graph.time_path[0] + ' is ' + query_graph.time_path[1] + '. '
                # cand_str += 'time is ' + query_graph.time_path[1] + '. '
                # print(item[1].lower())
                # print('time:', cand_str)
            cand_str += '\t'
            if(query_graph.ordinal_path != []):
                # cand_str += ' ' + ' '.join(query_graph.ordinal_path) + '.'
                # cand_str += query_graph.ordinal_path[0] + ' is ' + query_graph.ordinal_path[1] + '. '
                cand_str += query_graph.ordinal_path[0] + ' ' + query_graph.ordinal_path[1] + ' is ' + query_graph.ordinal_path[2] + '. '
                # cand_str += 'rank is ' + query_graph.ordinal_path[1] + '. '
                # print('ordinal:', cand_str)
                # cand_str += ' ' + query_graph.ordinal_path[0] + ' rank ' + query_graph.ordinal_path[1] + '.'
            cand_str += '\t'
            # ********************************************************************
            cand_str += ' '.join(query_graph.main_path[0:3]) + '.' # 包含主实体、关系和答案
            f.write(item[0] + '\t' + item[1].lower() + '\t' + cand_str.lower() + '\t' +\
                    str(query_graph.p) + '\t' + str(query_graph.r) + '\t' + str(query_graph.f1) + '\t' + item[3] + '\n')
    f.flush()
    print('问句个数：', num)


# 将数据写出到文件
def write2file_label_position_no_punc(file_name, query_answer):
    f = open(file_name, 'w', encoding='utf-8')
    num = 0
    for onequery_item in query_answer:
        num += 1
        for item in onequery_item:
            # import pdb; pdb.set_trace()
            query_graph = item[2]
            cand_str = ''
            # *********************约束路径放在主路径之前****************************
            if(query_graph.type_path != []):
                cand_str += ' '.join(query_graph.type_path)
                # 重复主路径，answer用类型代替
                # cand_str += ' '.join(query_graph.main_path[0:2]) + ' ' + query_graph.type_path[0] + '. '
            cand_str += '\t'
            if(query_graph.entity_path != []):
                cand_str += ' '.join(query_graph.entity_path)
                # cand_str += query_graph.entity_path[0] + ' is ' + query_graph.entity_path[1] + '. '
            cand_str += '\t'
            if(query_graph.time_path != []):
                cand_str += ' '.join(query_graph.time_path)
                # cand_str += query_graph.time_path[0] + ' is ' + query_graph.time_path[1] + '. '
                # cand_str += 'time is ' + query_graph.time_path[1] + '. '
                # print(item[1].lower())
                # print('time:', cand_str)
            cand_str += '\t'
            if(query_graph.ordinal_path != []):
                # cand_str += ' ' + ' '.join(query_graph.ordinal_path) + '.'
                # cand_str += query_graph.ordinal_path[0] + ' is ' + query_graph.ordinal_path[1] + '. '
                cand_str += query_graph.ordinal_path[0] + ' ' + query_graph.ordinal_path[1] + ' ' + query_graph.ordinal_path[2]
                # cand_str += 'rank is ' + query_graph.ordinal_path[1] + '. '
                # print('ordinal:', cand_str)
                # cand_str += ' ' + query_graph.ordinal_path[0] + ' rank ' + query_graph.ordinal_path[1] + '.'
            cand_str += '\t'
            # ********************************************************************
            cand_str += ' '.join(query_graph.main_path[0:3]) + '.' # 包含主实体、关系和答案
            f.write(item[0] + '\t' + item[1].lower() + '\t' + cand_str.lower() + '\t' +\
                    str(query_graph.p) + '\t' + str(query_graph.r) + '\t' + str(query_graph.f1) + '\t' + item[3] + '\n')
    f.flush()
    print('问句个数：', num)


def get_pos_neg_accord_f1_with_new_search(que2feature, qid2cands_improvement):
    qid2cand = {}
    for qid in que2feature:
        features = copy.deepcopy(que2feature[qid])
        # print(features)
        features.sort(key=takeF1, reverse=True) # 对一个问句对应的候选查询图进行排序
        # features = que2feature[qid]
        max_f1 = float(features[0].f1)
        flag = 0
        if(qid in qid2cands_improvement):
            query_graph = qid2cands_improvement[qid]
            if(max_f1 < query_graph.f1):
                # print(qid)
                # print('Luo:')
                # features[0].serialize()
                # print('重写：', query_graph.main_path, query_graph.entity_path, query_graph.time_path,
                #             query_graph.type_path, query_graph.ordinal_path, query_graph.p, query_graph.r, query_graph.f1)
                # print(qid, max_f1, query_graph.f1)
                # max_f1 = query_graph.f1
                flag = 1
                # import pdb; pdb.set_trace()
        for feature in features:
            if(qid not in qid2cand):
                qid2cand[qid] = [[], []]
                if(float(feature.f1) == max_f1 and max_f1 > 0.1):
                    qid2cand[qid][0].append(feature)# 0存放正确的候选
                # elif(float(feature.f1) == 0):
                #     qid2cand[qid][1].append(feature)# 1存放错误的候选
                else:
                    qid2cand[qid][1].append(feature)# 1存放错误的候选
            else:
                if(float(feature.f1) == max_f1 and max_f1 > 0.1):
                    qid2cand[qid][0].append(feature)# 0存放正确的候选
                # elif(float(feature.f1) == 0):
                #     qid2cand[qid][1].append(feature)# 1存放错误的候选
                else:
                    qid2cand[qid][1].append(feature)# 1存放错误的候选
        # if(int(qid) == 999):
        #     print(len(qid2cand[qid][0]), len(qid2cand[qid][1]))
        #     # print(qid2cand[qid])
        #     import pdb; pdb.set_trace()
        
        random.shuffle(qid2cand[qid][1], random=r_shuff)# 对负例进行随机化，排序baseline生成top的顺序影响
        random.shuffle(qid2cand[qid][0], random=r_shuff)
        if(flag == 1):
            query_graph = qid2cands_improvement[qid]
            if(query_graph.f1 > 0.1):
                qid2cand[qid][0].append(query_graph)
    return qid2cand


# 获取改进搜索之后的候选结果
def get_cands_from_improvement(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    qid2cands = {}
    lines = f.readlines()
    i = 0
    while(i < len(lines)):
        qid = lines[i].split('\t')[0]
        scores_list = lines[i + 1].strip().split('\t')
        p = float(scores_list[0])
        r = float(scores_list[1])
        f1 = float(scores_list[2])
        all_path = json.loads(lines[i + 3].strip())
        main_path = all_path[0]
        if(len(main_path) > 0):
            main_path[0] = normalize_entity(main_path[0])
            main_path[1] = normalize_relation(main_path[1])
            main_path[2] = normalize_answer(main_path[2])
        entity_path = all_path[1]
        if(len(entity_path) > 0):
            entity_path[0] = normalize_relation(entity_path[0])
            entity_path[1] = normalize_entity(entity_path[1])
        time_path = all_path[2]
        if(len(time_path) > 0):
            time_path[0] = normalize_relation(time_path[0])
        type_path = all_path[3]
        if(len(type_path) > 0):
            type_path[0] = normalize_relation(type_path[0])
            type_path[0] = 'type is ' + type_path[0]
        ordinal_path = all_path[4]
        if(len(ordinal_path) > 0):
            ordinal_path[0] = normalize_relation(ordinal_path[0])
        query_graph = QueryGraphForTrain(main_path = main_path, entity_path = entity_path, time_path = time_path,\
                                                        type_path = type_path, ordinal_path = ordinal_path, p = p, r = r, f1 = f1)
        # import pdb; pdb.set_trace()
        qid2cands[str(qid).zfill(4)] = query_graph
        i += 4
    return qid2cands




# if __name__ == "__main__":
#     init_dir_name = '../Generate_QueryGraph/Luo/runnings/candgen_CompQ/20201130_entity_time_type_ordinal/data/'
#     entity_dic = read_entity(init_dir_name)
#     qid2cands = read_query_graph(init_dir_name, entity_dic)
#     qid2cands = get_pos_neg_accord_f1(qid2cands)
#     f = open('/data2/yhjia/kbqa_sp/compq_qid2question.pkl', 'rb')
#     qid2question = pickle.load(f)
#     qid2cands_train, qid2cands_dev, qid2cands_test = split_data_compq(qid2cands)
#     file_name = './compq_classify_neg_' + str(NEG_NUM) + '_'
#     train_data = select_top_data(qid2question, qid2cands_train, pos_only1=False, data_type='v', neg=NEG_NUM)
#     write2file(file_name + 'train.txt', train_data)
#     # dev_data = select_top_data(qid2question, qid2cands_dev, pos_only1=False, data_type='v', neg=NEG_NUM)
#     # write2file(file_name + 'dev.txt', dev_data)
#     # test_data = select_top_data(qid2question, qid2cands_test, pos_only1=False, data_type='t', neg=NEG_NUM)
#     # write2file(file_name + 'test.txt', test_data)