# 从log文件中获取对应的结果
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def read_result(init_dir_name, feature):
    entity_dic = {}
    results = {}
    for root, dirs, files in os.walk(init_dir_name):
        # print(root, dirs, files)
        for dir_name in dirs: # 针对每组问句对应的文件夹进行处理
            # import pdb; pdb.set_trace()
            if(feature in dir_name):
                neg_num = dir_name.split('_')[-3]
                file_name = init_dir_name + dir_name + '/result.log'
                # print(file_name)
                f = open(file_name, 'r', encoding='utf-8')
                lines = f.readlines()
                # if(len(lines) > 0 and 'precision' in lines[-1]):
                if(len(lines) > 0 and '测试' in lines[-2]):
                    train = lines[-2].split('/')[-1].split('_')[-3]
                    dev = lines[-2].split('/')[-1].split('_')[-2]
                    prf = lines[-1].strip().split('\t')
                    p = prf[0].split(':')[-1]
                    r = prf[1].split(':')[-1]
                    f1 = prf[2].split(':')[-1]
                    if(neg_num not in results):
                        results[neg_num] = (train, dev, p, r, f1)
                    else:
                        print(file_name)
                        # import pdb; pdb.set_trace()
                        print('数据重复')
                else:
                    print(dir_name, '未完成')
    return results

# 将结果写成到文件中
def write2file(results, file_name):
    results_list = sorted(results.items(), key=lambda x: int(x[0]))
    f = open(file_name + '.txt', 'w', encoding='utf-8')
    for key in results_list:
        f.write('1:' + key[0] + '\n')
        f.write('%s%.4f%s%.4f%s%.4f\t%.4f\t%.4f'%('train:', round(float(key[1][0]), 3), '\tdev:', round(float(key[1][1]), 4), \
                '\ttest:', round(float(key[1][2]), 4), round(float(key[1][3]), 4), round(float(key[1][4]), 4)))
        f.write('\n')
    f.flush()


if __name__ == "__main__":
    # *******************WebQ****************************

    # features = []
    # features.append('bert_top5_seed_42_batch')
    # features.append('bert_listwise_from5531_top5_seed_42_batch')
    # features.append('bert_poscatneg_top5_seed_42_batch')
    # features.append('bert_pointwise_from5531_group5') # webq上的pointwise方法
    # for feature in features:
    #     results = read_result('./model/webq/', feature = feature)
    #     write2file(results, feature)


    # features = []
    # features.append('bert_pairwise_from5531_poscatneg')
    # features.append('bert_listwise_from5531_poscatneg')
    # features.append('bert_listwise_from5531_onecatone')
    # features.append('bert_pointwise_from5531_onecatone') # webq上的pointwise方法
    # for feature in features:
    #     results = read_result('./model/webq/', feature = feature)
    #     write2file(results, feature)

    # features = []
    # features.append('rerank_2bert_answer_type_bert_webq_pointwise_2scoreadd_neg_')
    # for feature in features:
    #     results = read_result(BASE_DIR + '/runnings/model/webq/', feature = feature)
    #     write2file(results, feature)

    # features = []
    # features.append('compq_resample_rank_pointwise_neg_')
    # for feature in features:
    #     results = read_result(BASE_DIR + '/runnings/model/compq/', feature = feature)
    #     write2file(results, feature)

    # features = []
    # # features.append('webq_rerank_pairwise_top_')
    # features.append('webq_rerank_2bert_answer_type_pairwise_to2add_neg_')
    # for feature in features:
    #     results = read_result(BASE_DIR + '/runnings/model/webq/pairwise/', feature = feature)
    #     write2file(results, feature)

    features = []
    # features.append('compq_rerank_pairwise_top_')
    features.append('compq_mulsigmoid_rerank_2bert_answer_type_pointwise_to2add_neg_')
    for feature in features:
        # results = read_result(BASE_DIR + '/runnings/model/compq/pairwise/', feature = feature)
        results = read_result(BASE_DIR + '/runnings/model/compq/', feature = feature)
        write2file(results, feature)
    