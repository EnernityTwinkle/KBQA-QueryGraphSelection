import os
import json
import csv
import sys


# 读取每个问句对应的实体链接表信息
def read_qid2entity(init_dir_name):
    '''
    功能：从第一阶段产生的link文件中提取每个问句qid对应的实体链接数据
    输入：第一阶段产生的候选文件夹名字
    输出：问句ID与链接到的实体构成的词典，其中key值为问句ID，value值为链接实体列表，形如：
    '''
    qid2entity = {}
    for root, dirs, files in os.walk(init_dir_name):
        # print(root, dirs, files)
        for dir_name in dirs: # 针对每组问句对应的文件夹进行处理
            file_names = os.listdir(root + dir_name)
            for file_name in file_names:
                if('_links' in file_name):
                    f = open(init_dir_name + dir_name + '/' + file_name, 'r', encoding = 'utf-8')
                    qid = file_name[0:4]
                    if(qid not in qid2entity):
                        qid2entity[qid] = []
                    for line in f:
                        line_json = json.loads(line.strip())
                        if(len(line_json) != 9):
                            print('长度不等于9')
                            import pdb; pdb.set_trace()
                        category = line_json[1][1]
                        if(line_json[6][0] != 'value'):
                            print('value 所在位置不一致')
                            import pdb; pdb.set_trace()
                        value = line_json[6][1]
                        if(line_json[7][0] != 'name'):
                            print('name 所在位置不一致')
                            import pdb; pdb.set_trace()
                        name = line_json[7][1]
                        qid2entity[qid].append((category + ' ' + value + ' ' + name))
                            # import pdb; pdb.set_trace()
    return qid2entity

def WebQ():
    result = {}
    init_dir_name = '../../runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    # init_dir_name = '/home/jiayonghui/github/bert_rank/runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    question2path = {}
    num = 0
    for root, dirs, files in os.walk(init_dir_name):
    #     print(root, dirs, files)
        for dir_name in dirs:
            file_names = os.listdir(init_dir_name + dir_name)
            for file_name in file_names:
    #             import pdb; pdb.set_trace()
                if('_schema' in file_name):
                    f = open(init_dir_name + dir_name + '/' + file_name, 'r', encoding = 'utf-8')
                    k = 0.0
                    temp = ''
                    mid = []
                    for line in f:
                        line_json = json.loads(line.strip())
    #                     import pdb; pdb.set_trace()
                        if(line_json['f1'] > k):
                            k = line_json['f1']
                            temp = line.strip()
                            for item in line_json['raw_paths']:
                                mid.append(item[3])
    #                     import pdb; pdb.set_trace()
                    que_id = (file_name[0:4])
                    # if(int(que_id) < 5810):
                    #     result[que_id + '\n' + temp] = k
                    if(int(que_id) < 3778):
                        result[que_id + '\n' + temp] = k
                    # if(int(que_id) >= 3778):
                    #     result[que_id + '\n' + temp] = k
                    question2path[que_id] = mid
    # print(result)
    print('len(result):', len(result))
    sum_f1 = 0.0
    f = open('./max_match_result.txt', 'w', encoding = 'utf-8')
    result_list = sorted(result.items(), key = lambda x:x[0])
    for item in result_list:
        if(item[1] <= 0.1):
            num += 1
        sum_f1 += item[1]
        f.write(item[0] + '\n')
    f.flush()
    print(sum_f1)
    print(sum_f1 / len(result))
    print('训练集中没有正确答案的问句个数：', num)
    # import pdb; pdb.set_trace()


def CompQ():
    result = {}
    init_dir_name = '/data2/yhjia/bert_rank/Generate_QueryGraph/Luo/runnings/candgen_CompQ/20201130_entity_time_type_ordinal/data/'
    for root, dirs, files in os.walk(init_dir_name):
        # print(root, dirs, files)
        for dir_name in dirs:
            file_names = os.listdir(init_dir_name + dir_name)
            for file_name in file_names:
    #             import pdb; pdb.set_trace()
                if('_schema' in file_name):
                    f = open(init_dir_name + dir_name + '/' + file_name, 'r', encoding = 'utf-8')
                    lines = f.readlines()
                    k = 0.0
                    temp = []
                    for line in lines:
                        line_json = json.loads(line.strip())
    #                     import pdb; pdb.set_trace()
                        if(line_json['f1'] > k):
                            k = line_json['f1']
                    if k > 0:
                        for line in lines:
                            line_json = json.loads(line.strip())
        #                     import pdb; pdb.set_trace()
                            if(line_json['f1'] == k):
                                temp.append(line.strip())
    #                     import pdb; pdb.set_trace()
                    que_id = (file_name[0:4])
                    # if(int(que_id) < 1300):
                    #     result[que_id + '\n' + '\n'.join(temp)] = k
                    if(int(que_id) < 3000):
                        result[que_id + '\n' + '\n'.join(temp)] = k
                    # if(int(que_id) < 3000 and int(que_id) >= 1300):
                    #     result[que_id + '\n' + '\n'.join(temp)] = k
    # print(result)
    print('len(result):', len(result))
    sum_f1 = 0.0
    f = open('./runnings/candgen_CompQ/max_match_result_joe.txt', 'w', encoding = 'utf-8')
    # f = open('./runnings/candgen_CompQ/max_match')
    result_list = sorted(result.items(), key = lambda x:x[0])
    for item in result_list:
        sum_f1 += item[1]
        # if(item[1] == 0.0):
        #     print(item)
        # import pdb; pdb.set_trace()
        f.write('\n'.join(item[0].split('\n')[0:2]) + '\n')
    f.flush()
    print(sum_f1)
    print(sum_f1 / len(result))
    # import pdb; pdb.set_trace()

def load_compq():
    compq_path = './qa-corpus/MulCQA'
    qa = []
    for Tvt in ('train', 'test'):
        fp = '%s/compQ.%s.release' % (compq_path, Tvt)
        print(fp)
        br = open(fp, 'r', encoding='utf-8')
        lines = br.readlines()
        for i, line in enumerate(lines):
            str_i = str(i).zfill(4)
            q, a_list_str = line.strip().split('\t')
            qa.append(q + '\n' + a_list_str)
    return qa

def load_webq():
    webq_path = './qa-corpus/web-question'
    qa_list = []
    for Tvt in ('train', 'test'):
        file_name = '%s/data/webquestions.examples.%s.json' % (webq_path, Tvt)
        fp = open(file_name, 'r', encoding='utf-8')
        webq_data = json.load(fp)
        # import pdb; pdb.set_trace()
        for raw_info in webq_data:
            qa = {}
            target_value = []
            ans_line = raw_info['targetValue']
            ans_line = ans_line[7: -2]      # remove '(list (' and '))'
            for ans_item in ans_line.split(') ('):
                ans_item = ans_item[12:]    # remove 'description '
                if ans_item.startswith('"') and ans_item.endswith('"'):
                    ans_item = ans_item[1: -1]
                target_value.append(ans_item)
            qa['utterance'] = raw_info['utterance']
            qa['targetValue'] = target_value
            qa_list.append(qa['utterance'] + '\n' + '\t'.join(qa['targetValue']))
    # import pdb; pdb.set_trace()
    # qa_list中每个元素格式:{'utterance': 'what is the name of justin bieber brother?', 'targetValue': ['Jazmyn Bieber', 'Jaxon Bieber']}
    return qa_list

# 获取在候选查询图中找不到正确答案的问句
def CompQ_no_answer():
    qa = load_compq()
    result = {}
    # init_dir_name = './runnings/candgen_CompQ/20200712_yh/data/'
    init_dir_name = '/data/yhjia/runnings/candgen_CompQ/200418_joe/data/'
    for root, dirs, files in os.walk(init_dir_name):
        print(root, dirs, files)
        for dir_name in dirs:
            file_names = os.listdir(init_dir_name + dir_name)
            for file_name in file_names:
    #             import pdb; pdb.set_trace()
                if('_schema' in file_name):
                    f = open(init_dir_name + dir_name + '/' + file_name, 'r', encoding = 'utf-8')
                    lines = f.readlines()
                    k = 0.0
                    temp = []
                    for line in lines:
                        line_json = json.loads(line.strip())
    #                     import pdb; pdb.set_trace()
                        if(line_json['f1'] > k):
                            k = line_json['f1']
                    # if(int(file_name[0:4]) == 619):
                    #     import pdb; pdb.set_trace()
                    if k <= 0: 
                        que_id = (file_name[0:4])
                        # if(int(que_id) >= 1300):
                        #     break
                        result[que_id] = qa[int(que_id)]
    # import pdb; pdb.set_trace()
    result_list = sorted(result.items(), key=lambda x: x[0])
    f = open('./runnings/candgen_CompQ/match_no_result_joe_2100.txt', 'w', encoding = 'utf-8')
    for item in result_list:
        f.write(item[0] + '\n' + item[1] + '\n')
    print('no answer 个数：', len(result_list))
    f.flush()


# 获取在候选查询图中找不到正确答案的问句
def WebQ_no_answer():
    qa = load_webq()
    result = {}
    
    # init_dir_name = '/data2/yhjia/kbqa_sp/runnings/candgen_WebQ/20200712_yh/data/'
    init_dir_name = '/data/yhjia/Question2Cands/runnings/candgen_WebQ/20201102_STAGG_add_answer_type_with_datatype/data/'
    qid2entity = read_qid2entity(init_dir_name)
    for root, dirs, files in os.walk(init_dir_name):
        print(root, dirs, files)
        for dir_name in dirs:
            file_names = os.listdir(init_dir_name + dir_name)
            for file_name in file_names:
    #             import pdb; pdb.set_trace()
                if('_schema' in file_name):
                    f = open(init_dir_name + dir_name + '/' + file_name, 'r', encoding = 'utf-8')
                    lines = f.readlines()
                    k = 0.0
                    temp = []
                    for line in lines:
                        line_json = json.loads(line.strip())
    #                     import pdb; pdb.set_trace()
                        if(line_json['f1'] > k):
                            k = line_json['f1']
                    if k <= 0.1: 
                        que_id = (file_name[0:4])
                        # if(int(que_id) >= 3778):
                        #     break
                        result[que_id] = (qa[int(que_id)], qid2entity[que_id])
    # import pdb; pdb.set_trace()
    result_list = sorted(result.items(), key=lambda x: x[0])
    f = open('./runnings/candgen_WebQ/match_no_result_stagg_add_answer_type_with_datatype.txt', 'w', encoding = 'utf-8')
    for item in result_list:
        f.write(item[0] + '\n' + ' ## '.join(item[1][1]) + '\n' + item[1][0] + '\n')
    f.flush()
    print('no answer 个数：', len(result_list))


if __name__ == "__main__":
    #**************获取WebQ数据集查询图生成模块的平均F1性能*******************
    WebQ()
    #**************获取CompQ数据集查询图生成模块的平均F1性能*******************
    # CompQ()
    #*********************************************************************
    # CompQ_no_answer()
    # WebQ_no_answer()