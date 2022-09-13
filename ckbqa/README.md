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


# ccks2021Plus数据集
## 构建数据
```
cd build_dataset
python trans_to_datasets.py
```

## 数据划分:训练集、验证集、测试集
```
cd question_classification/build_data
python build_query_graph_data.py
```

## 查询图生成
```
cd build_query_graph
python main_based_filter_rel_for_complex_que.py    #训练集和验证集
python main_based_filter_rel_for_test.py    #测试集
```

## 查询图转序列
```
cd src/querygraph2seq
python querygraph_to_seq.py
```

## 训练数据构建
```
cd src/build_model_data
python build_train_data_for_analysis.py
python build_test_data.py
```

## 模型训练
```
cd src/model_train
python train_listwise_multi_types_1.py
```
