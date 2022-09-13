from curses.panel import top_panel
import sys
import os
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


# 从已经拍好序的候选文件中选出top_n构建重排序数据
def get_top_n(file_name, top_n):
    f = open(file_name, 'r', encoding='utf-8')
    qid2data = {}
    lines = f.readlines()
    for line in lines:
        line_list = line.strip().split('\t')
        # import pdb; pdb.set_trace()
        qid = eval(line_list[0])[0]
        if(qid not in qid2data):
            temp = []
            temp.append((line_list[0]))
            qid2data[qid] = temp
        else:
            if(len(qid2data[qid]) < top_n):
                qid2data[qid].append((line_list[0]))
    # import pdb; pdb.set_trace()
    return qid2data

# 将数据写出到文件中，针对'T'/'v'/'t'会有不同的操作
def write2file(file_name, query_answer, data_type):
    f = open(file_name, 'w', encoding='utf-8')
    if(data_type == 'T'):
        for qid in query_answer:
            num = 0
            length = len(query_answer[qid])
            for item in query_answer[qid]:
                # import pdb; pdb.set_trace()
                f.write(item + '\n')
                num += 1
            while(num < top_n):
                f.write(query_answer[qid][num % length] + '\n')
                num += 1
    elif(data_type == 't' or data_type == 'v'):
        for qid in query_answer:
            # import pdb; pdb.set_trace()
            for item in query_answer[qid]:
                f.write(item + '\n')
    f.flush()
        

if __name__ == "__main__":
    # dev和test用这个
    ########## CCKS2021-CompKBQA
    dirName = BASE_DIR + '/data/train_data/ccks_comp/'
    for top_n in [5, 10, 20, 30, 40, 50, 60, 80, 100]:
        print(top_n)
        file_name = dirName + '7490_ccks_comp_listwise_bert_devneg100_seq150_CE_group_20_1_42_100_test_from.txt'
        qid2data = get_top_n(file_name, top_n)
        write2file(dirName + 'rerank_t_bert_top' + str(top_n) + '.txt', qid2data, 'T')

        file_name = dirName + '7490_ccks_comp_listwise_bert_devneg100_seq150_CE_group_20_1_42_100_dev_from.txt'
        qid2data = get_top_n(file_name, top_n)
        write2file(dirName + 'rerank_v_bert_top' + str(top_n) + '.txt', qid2data, 'T')

