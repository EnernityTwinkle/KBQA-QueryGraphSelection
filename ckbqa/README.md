# CKBQA

这里是针对中文领域的KBQA系统，知识库采用PKUBASE。


## 数据格式

* 训练集格式参考文件
/dataset/ccks2021/ccks2021_task13_train.txt
* 实体识别结果格式参考文件
/data/entitylinking/1031_EL_train.json 

将data/models里的文件放到对应位置

torch版本为1.0.0


## 错误分析

通过src/eda/multiType_error_analysis.py分析每种类型问句的f1值。

# 结果复现

## 排序前置结果

### 查询图生成

非必要, 已提供查询图生成结果

```bash
cd /src/build_query_graph
python main_based_filter_rel_for_complex_que.py    #训练集和验证集
python main_based_filter_rel_for_test.py    #测试集
```

评估查询图生成性能

```bash
cd /src/build_query_graph
bash eval_max_f1.sh

# 仅对CCKS2021-test(比赛放未公开数据, 本团队标注)
bash eval_max_f1_test.sh
```

### 查询图转序列

非必要, 已提供转换结果

```bash
cd src/querygraph2seq
python querygraph_to_seq.py
```

### 构建stage1排序输入

非必要, 已提供结果

* CCKS2019
* CCKS2021

```bash
cd src/build_model_data
python build_train_data_for_analysis.py # 构建训练数据
python build_test_data.py   # 转化test
```

## stage1模型训练

```bash
cd src/model_train
python train_listwise_multi_types_1.py
```

我们同样提供已训练的模型

* CCKS2019
* CCKS2021

## stage2模型训练

```bash
```
