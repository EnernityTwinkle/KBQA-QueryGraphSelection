# Two-stage Query Graph Selection for Knowledge Base Question Answering

代码 文档还在整理中, 目前上传的还是中间版本。后续会新增EN-readme

更多细节可参考 __Yonghui Jia, Chuanyuan Tan, Yuehe Chen, Muhua Zhu, Pingfu Chao, Wenliang Chen. Two-stage Query Graph Selection for Knowledge Base Question Answering. NLPCC 2022.__

中文数据集上的实验参考ckbqa/README.md 展开(待上传)

## 生成查询图模块

1、给定问句，生成查询图候选，首先基于Luo方法中的生成方法(代码模块Generate_QueryGraph/Luo)

### ComplexQuestions
可得到CompQ数据集对应的候选查询图
```
cd Generate_QueryGraph/Luo
bash step1_gen_query_graph_compq_luo.sh
```

评价得到的候选查询图的平均召回率，即每个问句对应最高f1值的平均(Generate_QueryGraph/Luo/max_f1.py)：
训练集和验证集（0.6333），测试集（0.6304）,整个数据集平均（0.6322）;
整个数据集上每个问句对应的平均候选个数为208(Generate_QueryGraph/Luo/build_listwise_data.py)

### WebQuestions
得到WebQ数据集对应的候选查询图
```
cd Generate_QueryGraph/Luo
bash step1_gen_query_graph_webq_luo.sh
```

评价得到的候选查询图的平均召回率，即每个问句对应最高f1值的平均(Generate_QueryGraph/Luo/max_f1.py)：
训练集和验证集（0.7852），测试集（0.7772）,整个数据集平均（0.7824）;
整个数据集上每个问句对应的平均候选个数为170(Generate_QueryGraph/Luo/WebQ/build_listwise_data.py)

### ping

2、重写的多模式搜索的查询图生成方法（Generate_QueryGraph/Question2Cands）

### ComplexQuestions

暴力搜索得到所有相关的一跳或两跳路径

对暴力搜索得到的路径进行子图重建（Generate_QueryGraph/Question2Cands/backward_search/step2_select_path.sh）


### WebQuestions

暴力搜索得到所有相关的一跳或两跳路径(yhjia@192.168.126.124:/data/yhjia/Question2Cands/backward_search/step1_webq_query_graph_with_linkings_and_answer.sh)


## 查询图选择模块

### 从生成的查询图候选到训练测试数据的构建(Build_Data)

#### WebQ

```
cd Build_Data/WebQ
python build_listwise_data.py
```
执行后会在runnings/train_data/webq文件夹下生成不同正负例的训练集文件，以及统一的验证和测试文件。这些文件会被三种排序优化方法共同使用。

单点排序训练数据：read_query_graph.py

805条    300条    799条


### 重排序
```
cd Model/prerank/pairwise/webq
python predict_dev_data_webq.py     根据训练好的排序模型计算验证集候选的得分
python predict_train_data_webq.py   根据训练好的排序模型计算训练集候选的得分
```