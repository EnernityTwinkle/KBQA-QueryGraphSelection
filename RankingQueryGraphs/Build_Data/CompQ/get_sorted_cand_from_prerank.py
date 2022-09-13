import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


# 对重复生成5次的训练集得分进行处理
def get_scores_train(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    new_file_name = file_name + '518484'
    fout = open(new_file_name, 'w', encoding='utf-8')
    lines = f.readlines()
    i = 0
    while(i < len(lines)):
        fout.write(lines[i])
        i += 5
    fout.flush()


# 根据得分将所有候选进行排序并输出
def sort_cands(scores_file_name, data_file_name, dataType):
    # sorted_file_name = data_file_name.replace('.txt', '_sorted.txt')
    # f_sorted = open(sorted_file_name, 'w', encoding='utf-8')
    modelName = scores_file_name.split('/')[-2]
    sorted_file_name = '/'.join(data_file_name.split('/')[0:-1]) + '/' + modelName + '_' + dataType + '.txt' 
    # import pdb; pdb.set_trace()
    f_sorted = open(sorted_file_name, 'w', encoding='utf-8')
    f = open(scores_file_name, 'r', encoding='utf-8')
    lines_predict = f.readlines()
    f2 = open(data_file_name, 'r', encoding='utf-8')
    lines = f2.readlines()
    begin = -1
    end = -1
    qid = ''
    i = 0
    sum_f1 = 0.0
    while i < len(lines):
        line = lines[i]
        qid_temp = line.strip().split('\t')[-1]
        if(qid_temp != qid):
            qid = qid_temp
            if(begin == -1):
                begin = i
            else:
                end = i
        if(end != -1):
            scores_lines = lines_predict[begin:end]
            scores = {}
            for item_i, item in enumerate(scores_lines):
                scores[item_i] = float(item.strip())
            scores_list = sorted(scores.items(), key=lambda x:x[1], reverse=True)
            for item in scores_list:
                f_sorted.write(lines[begin+item[0]].strip() + '\t' + str(item[1]) + '\n')
            # num = len(scores) - 1
            begin = -1
            end = -1
            qid = ''
            i -= 1
            # import pdb; pdb.set_trace()
        i += 1
    if(begin != -1):
        scores_lines = lines_predict[begin:]
        scores = {}
        for item_i, item in enumerate(scores_lines):
            scores[item_i] = float(item.strip())
        scores_list = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        for item in scores_list:
            f_sorted.write(lines[begin+item[0]].strip() + '\t' + str(item[1]) + '\n')
    print('数据个数：', len(lines))

if __name__ == "__main__":
    # sort_cands(BASE_DIR + '/runnings/train_data/compq/compq_2cv_from_3835_scores.txt', 
    #         BASE_DIR + '/runnings/train_data/compq/compq_train_all.txt', 'train_2cv')
    # sort_cands(BASE_DIR + '/runnings/model/compq/bert_compq_pointwise_3835/prediction_valid', 
    #         BASE_DIR + '/runnings/train_data/compq/compq_dev_all.txt', 'dev')
    # sort_cands(BASE_DIR + '/runnings/model/compq/bert_compq_pointwise_3835/prediction', 
    #         BASE_DIR + '/runnings/train_data/compq/compq_test_all.txt', 'test')

    # sort_cands(BASE_DIR + '/runnings/model/compq/3711_bert_compq_pointwise_withans_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_20_42_100/prediction_train', 
    #         BASE_DIR + '/runnings/train_data/compq/compq_train_all.txt', 'train_sorted')
    # sort_cands(BASE_DIR + '/runnings/model/compq/3711_bert_compq_pointwise_withans_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_20_42_100/prediction_dev', 
    #         BASE_DIR + '/runnings/train_data/compq/compq_dev_all.txt', 'dev_sorted')
    # sort_cands(BASE_DIR + '/runnings/model/compq/3711_bert_compq_pointwise_withans_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_20_42_100/prediction_test', 
    #         BASE_DIR + '/runnings/train_data/compq/compq_test_all.txt', 'test_sorted')

    sort_cands(BASE_DIR + '/runnings/model/compq/4454_new_bert_compq_listwise_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_120_42_100/prediction_test', 
            BASE_DIR + '/runnings/train_data/compq/compq_test_all.txt', 'test_sorted_4440')