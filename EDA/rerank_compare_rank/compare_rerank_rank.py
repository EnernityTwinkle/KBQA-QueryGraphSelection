import sys
import os
import json
from typing import List, Dict, Tuple
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


# 读取预测的得分文件
def read_prediction_scores(file_name: str) -> List[float]:
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    scores: List[float] = []
    for i, line in enumerate(lines):
        scores.append(float(line.strip()))
    return scores


# 读取候选文件
def read_cands(file_name: str, qidIndex = -1):
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    qid2cands = {}# 记录每个问句对应的候选信息
    qid2pos = {}# 记录每个问句对应候选的开始和结束位置
    qid_list = [] # 记录问句ID
    before_qid = ''
    current_qid = ''
    for i, line in enumerate(lines):
        line_cut = line.strip().split('\t')
        current_qid = line_cut[qidIndex]
        if(current_qid not in qid2cands):
            qid2cands[current_qid] = []
            qid2cands[current_qid].append(line.strip())
            qid2pos[current_qid] = []
            qid2pos[current_qid].append(i) # 记录开始的位置
            qid_list.append(current_qid)
            if(before_qid != ''):
                qid2pos[before_qid].append(i - 1) # 记录结束的位置
            before_qid = current_qid
        else:
            qid2cands[current_qid].append(line.strip())
    if(len(qid2pos[before_qid]) == 1):
        qid2pos[before_qid].append(i)
        print('before_qid:', before_qid, qid2pos[before_qid])
    # import pdb; pdb.set_trace()
    return qid2cands, qid2pos, qid_list


# 针对每个问句的候选进行排序，并输出选择出错的问句信息
def sorted_and_print(qid2cands, qid2pos, qid_list, scores):
    fwrite = open('./classify_5583_error_que.txt', 'w', encoding='utf-8')
    error_num = 0
    for qid in qid_list:
        cands = qid2cands[qid]
        pos = qid2pos[qid]
        current_scores = scores[pos[0]: pos[1] + 1]
        sorted_scores = sorted(enumerate(current_scores), key=lambda x: x[1], reverse=True)
        if(cands[sorted_scores[0][0]][0] == '0'):
            error_num += 1
            for item in sorted_scores:
                fwrite.write(cands[item[0]] + '\n')
                # import pdb; pdb.set_trace()
            fwrite.write('\n')
        # import pdb; pdb.set_trace()
    fwrite.flush()
    print('回答错误的问句数量：', error_num)


# 计算每个问句的f1值得分,根据模型预测得分进行排序选取
def get_qid2f1(qid2cands, qid2pos, qid_list, scores, f1Pos):
    qid2f1 = {}
    qid2onecand = {}
    for qid in qid_list:
        cands = qid2cands[qid]
        pos = qid2pos[qid]
        current_scores = scores[pos[0]: pos[1] + 1]
        sorted_scores = sorted(enumerate(current_scores), key=lambda x: x[1], reverse=True)
        # import pdb; pdb.set_trace()
        try:
            # import pdb; pdb.set_trace()
            qid2f1[qid] = float(cands[sorted_scores[0][0]].split('\t')[f1Pos])
            qid2onecand[qid] = cands[sorted_scores[0][0]]
        except:
            import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
    return qid2f1, qid2onecand

# 计算每个问句对应的f1值上限得分,根据查询图f1值得分进行排序选取
def get_qid2maxf1(qid2cands, qid2pos, qid_list):
    qid2f1 = {}
    qid2maxcand = {}
    for qid in qid_list:
        cands = qid2cands[qid]
        pos = qid2pos[qid]
        cands.sort(key=lambda x:float(x.split('\t')[-2]), reverse=True)
        # import pdb; pdb.set_trace()
        try:
            # import pdb; pdb.set_trace()
            qid2f1[qid] = float(cands[0].split('\t')[-2])
            qid2maxcand[qid] = cands[0]
        except:
            import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
    return qid2f1, qid2maxcand

    
def load_compq():
    compq_path = BASE_DIR + '/qa-corpus/MulCQA'
    qa_list = []
    for Tvt in ('train', 'test'):
        fp = '%s/compQ.%s.release' % (compq_path, Tvt)
        f = open(fp, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            qa = {}
            q, a_list_str = line.strip().split('\t')
            qa['utterance'] = q
            qa['targetValue'] = json.loads(a_list_str)
            qa_list.append(qa)
    print('%d CompQuesetions loaded.', len(qa_list))
    return qa_list
    
def load_webq():
    '''
    功能：读取webq数据集
    '''
    webq_path = BASE_DIR + '/qa-corpus/web-question'
    qa_list = []
    for Tvt in ('train', 'test'):
        webq_fp = webq_path + '/data/webquestions.examples.' + Tvt + '.json'
        print(webq_fp)
        f = open(webq_fp, 'r', encoding='utf-8')
        webq_data = json.load(f)
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
            qa_list.append(qa)
            
    # import pdb; pdb.set_trace()
    # qa_list中每个元素格式:{'utterance': 'what is the name of justin bieber brother?', 'targetValue': ['Jazmyn Bieber', 'Jaxon Bieber']}
    print('%d WebQuesetions loaded.', len(qa_list))
    return qa_list


# 分析查询图构建成功，而排序模型出错的情况
def sort_error_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand):
    qid2diff = {}
    for qid in qid2maxf1:
        if(qid2maxf1[qid] > qid2f1[qid] and qid2maxf1[qid] > 0.1):
            qid2diff[qid] = (qid2maxcand[qid], qid2onecand[qid])
    return qid2diff


def true_false(qid2f1_true, qid2onecand_true, qid2f1_false, qid2onecand_false, f):
    for qid in qid2f1_true:
        if(qid2f1_true[qid] > qid2f1_false[qid] and qid2f1_true[qid] > 0.1 and qid2f1_false[qid] < 0.1):
            f.write(qid2onecand_true[qid] + '\n')
            f.write(qid2onecand_false[qid] + '\n')
            # import pdb; pdb.set_trace()
            # print(qid2onecand_true[qid])
            # print(qid2onecand_false[qid])

def false_true(qid2f1_true, qid2onecand_true, qid2f1_false, qid2onecand_false, f):
    for qid in qid2f1_true:
        if(qid2f1_true[qid] < qid2f1_false[qid] and qid2f1_false[qid] > 0.1 and qid2f1_true[qid] < 0.1):
            f.write(qid2onecand_true[qid] + '\n')
            f.write(qid2onecand_false[qid] + '\n')


def webq_compare():
    file_name = './prediction_test_webq_top20.txt'
    file_name_pointwise = './prediction_test_webq_all.txt'
    scores = read_prediction_scores(file_name)
    scores_point = read_prediction_scores(file_name_pointwise)
    file_name = BASE_DIR + '/runnings/train_data/webq/webq_t_top20_from5244.txt'
    file_nameAll = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    qid2cands, qid2pos, qid_list = read_cands(file_name, -2)
    qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, scores, -3)
    qid2candsAll, qid2posAll, qid_listAll = read_cands(file_nameAll, -1)
    qid2f1_point, qid2onecand_point = get_qid2f1(qid2candsAll, qid2posAll, qid_listAll, scores_point, -2)
    qid2maxf1, qid2maxcand = get_qid2maxf1(qid2cands, qid2pos, qid_list)
    f = open('webq_rerank_true_rank_false.txt', 'w', encoding='utf-8')
    true_false(qid2f1, qid2onecand, qid2f1_point, qid2onecand_point, f)
    f = open('webq_listwise_false_pointwise_true.txt', 'w', encoding='utf-8')
    false_true(qid2f1, qid2onecand, qid2f1_point, qid2onecand_point, f)
    import pdb; pdb.set_trace()
    # 获取qid与问句之间的对应关系
    # qa_list = load_compq()
    qa_list = load_webq()
    que2qid = {}
    for i, qa in enumerate(qa_list):
        que2qid[qa['utterance']] = i
    # f = open('/data/yhjia/cytan/train2.0.txt', 'r', encoding='utf-8')
    # f = open('/data2/yhjia/cytan/complex_test.txt', 'r', encoding='utf-8')
    f = open('/data2/yhjia/cytan/web_test.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    i = 0
    sum_f1 = 0.0
    num = 0.0
    # for que in que_dic:
    for que in que2qid:
        qid = que2qid[que]
        # import pdb; pdb.set_trace()
        if(str(qid).zfill(4) in qid2f1):
            sum_f1 += qid2f1[str(qid).zfill(4)]
            num += 1
    print(sum_f1, num, sum_f1 / num)
    qid2diff = sort_error_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand)
    fout = open('./查询图.txt', 'w', encoding='utf-8')
    for qid in qid2diff:
        fout.write(qid2diff[qid][0] + '\n')
        fout.write(qid2diff[qid][1] + '\n')
    fout.flush()
    import pdb; pdb.set_trace()


def compq_compare():
    file_name = './prediction_test_compq_top40.txt'
    file_name_pointwise = './prediction_test_compq_all.txt'
    scores = read_prediction_scores(file_name)
    scores_point = read_prediction_scores(file_name_pointwise)
    file_name = BASE_DIR + '/runnings/train_data/compq/compq_t_top40.txt'
    fileNameRank = BASE_DIR + '/runnings/train_data/compq/compq_rank1_f01_gradual_label_position_listwise_1_140_type_entity_time_ordinal_mainpath_test_all.txt'
    qid2cands, qid2pos, qid_list = read_cands(file_name, -2)
    qid2candsAll, qid2posAll, qid_listAll = read_cands(fileNameRank, -1)
    qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, scores, -3)
    qid2f1_point, qid2onecand_point = get_qid2f1(qid2candsAll, qid2posAll, qid_listAll, scores_point, -2)
    qid2maxf1, qid2maxcand = get_qid2maxf1(qid2cands, qid2pos, qid_list)
    f = open('compq_rerank_true_rank_false.txt', 'w', encoding='utf-8')
    true_false(qid2f1, qid2onecand, qid2f1_point, qid2onecand_point, f)
    f = open('compq_listwise_false_pointwise_true.txt', 'w', encoding='utf-8')
    false_true(qid2f1, qid2onecand, qid2f1_point, qid2onecand_point, f)
    import pdb; pdb.set_trace()
    # 获取qid与问句之间的对应关系
    # qa_list = load_compq()
    qa_list = load_webq()
    que2qid = {}
    for i, qa in enumerate(qa_list):
        que2qid[qa['utterance']] = i
    # f = open('/data/yhjia/cytan/train2.0.txt', 'r', encoding='utf-8')
    # f = open('/data2/yhjia/cytan/complex_test.txt', 'r', encoding='utf-8')
    f = open('/data2/yhjia/cytan/web_test.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    i = 0
    sum_f1 = 0.0
    num = 0.0
    # for que in que_dic:
    for que in que2qid:
        qid = que2qid[que]
        # import pdb; pdb.set_trace()
        if(str(qid).zfill(4) in qid2f1):
            sum_f1 += qid2f1[str(qid).zfill(4)]
            num += 1
    print(sum_f1, num, sum_f1 / num)
    qid2diff = sort_error_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand)
    fout = open('./查询图.txt', 'w', encoding='utf-8')
    for qid in qid2diff:
        fout.write(qid2diff[qid][0] + '\n')
        fout.write(qid2diff[qid][1] + '\n')
    fout.flush()
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    webq_compare()
    # compq_compare()
    
