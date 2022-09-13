
# 计算预测文件与gold文件的得分
def cal_f1(file_name1, file_name2, data_type, actual_num = 0, log=0):
    f = open(file_name1, 'r', encoding='utf-8')
    lines_predict = f.readlines()
    f2 = open(file_name2, 'r', encoding='utf-8')
    lines = f2.readlines()
    assert len(lines) == len(lines_predict)
    begin = -1
    end = -1
    qid = ''
    i = 0
    sum_f1 = 0.0
    have_p_r = 0
    p = 0.0
    r = 0.0
    que_num = 0
    while i < len(lines):
        line_cut = lines[i].strip().split('\t')
        have_p_r = 1
        line = lines[i]
        qid_temp = line.strip().split('\t')[-1]
        if(qid_temp != qid):
            qid = qid_temp
            if(begin == -1):
                begin = i
            else:
                end = i
        if(end != -1):
            scores = lines_predict[begin:end]
            max_score = -100000000
            num = 0
            for j, score in enumerate(scores):
                if(float(score.strip()) >= max_score):
                    max_score = float(score.strip())
                    num = j
            # num = len(scores) - 1
            f1 = float(lines[begin + num].split('\t')[-2])
            if(have_p_r):
                p += float(lines[begin + num].split('\t')[-4])
                r += float(lines[begin + num].split('\t')[-3])
            if(log == 1):
                print(p, r, f1)
            # if(f1 > 0):
            #     print(begin+num, f1)
            sum_f1 += f1
            que_num += 1
            # print(begin, end, num, f1)
            # import pdb; pdb.set_trace()
            begin = -1
            end = -1
            qid = ''
            i -= 1
            # import pdb; pdb.set_trace()
        i += 1
    if(begin != -1):
        # import pdb; pdb.set_trace()
        scores = lines_predict[begin:]
        max_score = -100000000
        num = 0
        for j, score in enumerate(scores):
            if(float(score.strip()) >= max_score):
                max_score = float(score.strip())
                num = j
        # num = len(scores) - 1
        f1 = float(lines[begin + num].split('\t')[-2])
        if(have_p_r):
            p += float(lines[begin + num].split('\t')[-4])
            r += float(lines[begin + num].split('\t')[-3])
        if(log == 1):
            print(p, r, f1)
        sum_f1 += f1
        que_num += 1
    print('数据个数：', len(lines))
    print('问句个数：', que_num)
    if(actual_num == 0):
        if(data_type == 't'):
            print(sum_f1, p / 2032.0, r / 2032.0, sum_f1 / 2032.0)
            return p / 2032.0, r / 2032.0, sum_f1 / 2032.0
        else:
            print(sum_f1, p / 755.0, r / 755.0, sum_f1 / 755.0)
            return p / 755.0, r / 755.0, sum_f1 / 755.0
    else:
        if(data_type == 't'):
            print(sum_f1, p / 800.0, r / 800.0, sum_f1 / 800.0)
            return p / 800.0, r / 800.0, sum_f1 / 800.0
        else:
            print(sum_f1, p / 300.0, r / 300.0, sum_f1 / 300.0)
            return p / 300.0, r / 300.0, sum_f1 / 300.0


# 计算预测文件与gold文件的得分
def cal_f1_with_scores(file_name1, file_name2, data_type):
    f = open(file_name1, 'r', encoding='utf-8')
    lines_predict = f.readlines()
    f2 = open(file_name2, 'r', encoding='utf-8')
    lines = f2.readlines()
    print(len(lines), len(lines_predict))
    assert len(lines) == len(lines_predict)
    begin = -1
    end = -1
    qid = ''
    i = 0
    have_p_r = 0
    sum_f1 = 0.0
    p = 0.0
    r = 0.0
    que_num = 0
    while i < len(lines):
        line_cut = lines[i].strip().split('\t')
        if(len(line_cut) == 11):
            have_p_r = 1
        line = lines[i]
        qid_temp = line.strip().split('\t')[-2]
        if(qid_temp != qid):
            qid = qid_temp
            if(begin == -1):
                begin = i
            else:
                end = i
        if(end != -1):
            scores = lines_predict[begin:end]
            max_score = -100000000
            num = 0
            for j, score in enumerate(scores):
                if(float(score.strip()) >= max_score):
                    max_score = float(score.strip())
                    num = j
            # num = len(scores) - 1
            f1 = float(lines[begin + num].split('\t')[-3])
            if(have_p_r):
                p += float(lines[begin + num].split('\t')[-5])
                r += float(lines[begin + num].split('\t')[-4])
            # if(f1 > 0):
            #     print(begin+num, f1)
            sum_f1 += f1
            que_num += 1
            # print(begin, end, num, f1)
            # import pdb; pdb.set_trace()
            begin = -1
            end = -1
            qid = ''
            i -= 1
            # import pdb; pdb.set_trace()
        i += 1
    if(begin != -1):
        # import pdb; pdb.set_trace()
        scores = lines_predict[begin:]
        max_score = -100000000
        num = 0
        for j, score in enumerate(scores):
            if(float(score.strip()) >= max_score):
                max_score = float(score.strip())
                num = j
        # num = len(scores) - 1
        f1 = float(lines[begin + num].split('\t')[-3])
        if(have_p_r):
            p += float(lines[begin + num].split('\t')[-5])
            r += float(lines[begin + num].split('\t')[-4])
        sum_f1 += f1
        que_num += 1
    print('数据个数：', len(lines))
    print('问句个数：', que_num)
    if(data_type == 't'):
        print(sum_f1, p / 2032.0, r / 2032.0, sum_f1 / 2032.0)
        return sum_f1 / 2032.0
    else:
        print(sum_f1, p / 755.0, r / 755.0, sum_f1 / 755.0)
        return sum_f1 / 755.0


if __name__ == "__main__":
    file_name1 = './bert_pairwise_all_3level_min20_1_49_2_400/prediction'
    file_name2 = './t_bert_rel_answer_pairwise_1_600000.txt'
    cal_f1(file_name1, file_name2, 't', actual_num=0)
