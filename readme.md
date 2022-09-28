# Two-stage Query Graph Selection for Knowledge Base Question Answering

后续会新增EN-readme

更多细节可参考 __Yonghui Jia, Chuanyuan Tan, Yuehe Chen, Muhua Zhu, Pingfu Chao, Wenliang Chen. Two-stage Query Graph Selection for Knowledge Base Question Answering. NLPCC 2022.__

中文数据集上的实验参考ckbqa/README.md 展开(按issues要求, 后续会进一步完善)

## 💾 数据获取

- [CCKS2019-KBQA](https://www.biendata.xyz/competition/ccks_2019_6/) 源自CCKS2019评测任务: 中文知识图谱问答

- [CCKS2021-KBQA](https://www.biendata.xyz/competition/ccks_2021_ckbqa/) 源自CCKS2021评测任务: 生活服务知识图谱问答评测

- [WebQuestions](https://nlp.stanford.edu/software/sempre/) 源自论文[Semantic Parsing on Freebase from Question-Answer Pairs](https://aclanthology.org/D13-1160/)

- [ComplexQuestions](https://github.com/JunweiBao/MulCQA/tree/ComplexQuestions) 源自论文[Constraint-Based Question Answering with Knowledge Graph](https://aclanthology.org/C16-1236.pdf)

- 我们已训练的模型和部分中间结果,包含中文和英文数据集内容 [百度网盘](https://pan.baidu.com/s/198gZPkUDPmoMEFJV0IKwoA?pwd=h35j)

## 🚀 快速复现实验结果

此处介绍如何复现英文数据集上的实验结果，关于中文部分请移步[ckbqa](https://github.com/cytan17726/KBQA-QueryGraphSelection/tree/master/ckbqa)

### 1️⃣ 查询图生成

非必要, 我们提供各数据集的生成结果，供排序使用

- WebQ: `RankingQueryGraphs/runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data`
- CompQ: `RankingQueryGraphs/runnings/candgen_CompQ/20201130_entity_time_type_ordinal/data`

```bash
cd Generate_QueryGraph/Luo
bash step1_gen_query_graph_webq_luo.sh
# 评价得到的候选查询图的平均召回率，即每个问句对应最高f1值的平均(Generate_QueryGraph/Luo/max_f1.py)：
# 训练集和验证集（0.7852），测试集（0.7772）,整个数据集平均（0.7824）
```

- 生成CompQ数据集对应的候选查询图
  - 已生成数据目录: /runnings/candgen_CompQ/20201130_entity_time_type_ordinal/data

```bash
cd Generate_QueryGraph/Luo
bash step1_gen_query_graph_compq_luo.sh
# 评价得到的候选查询图的平均召回率，即每个问句对应最高f1值的平均(Generate_QueryGraph/Luo/max_f1.py)：
# 训练集和验证集（0.6333），测试集（0.6304）,整个数据集平均（0.6322）
```

### 2️⃣ 构建stage1 排序的输入数据

- WebQ

```bash
cd Build_Data/WebQ
python build_listwise_data.py
```

- CompQ

```bash
cd Build_Data/CompQ/
python build_listwise_data.py
```

### 3️⃣ stage1 排序

- 我们提供已训练好的模型 `RankingQueryGraphs/runnings/model`

```bash
cd Model/Listwise
# WebQ
python main_bert_listwise_webq.py
# CompQ
python main_bert_listwise_comp.py
# 需要修改对应参数
```

### 4️⃣ 构建stage2 排序的输入数据

- WebQ

```bash
cd Build_Data/WebQWebQ
# 获得 初排得分
python get_sorted_cand_from_prerank_score.py
# 选取初排得分Topn(用于dev和test)
python selet_topn_from_sorted.py
# 选取n个负例(用于train)
python select_1_n.py
```

- CompQ

```bash
cd CompQ
# 获得 初排得分
python get_sorted_cand_from_prerank.py
# 选取初排得分Topn(用于dev和test)
python selet_topn_from_sorted.py
# 选取n个负例(用于train)
python select_1_n.py
```

### 5️⃣ 重排序

#### 基于stage1模型对所有候选打分

``` bash
# 基于stage1模型，对train, dev, test全部候选打分
# webq
cd Model/prerank/webq
python predict_test_data_webq.py
python predict_dev_data_webq.py
python predict_train_data_webq.py
# compq
cd Model/prerank/compq
python predict_test_data_compq.py
python predict_dev_data_compq.py
python predict_train_data_compq.py
```

#### 根据排序得分获取有序的候选查询图

```bash
# 需要修改参数
# webq
cd Build_Data/WebQ
python get_sorted_cand_from_prerank_score.py
# comq
cd Build_Data/CompQ
python get_sorted_cand_from_prerank.py
```

#### 根据有序的候选查询图构建重排序数据

```bash
# 需要修改参数
# webq
cd Build_Data/WebQ
python select_1_n.py
python select_topn_from_sorted.py
# compq
cd Build_Data/CompQ
python select_1_n.py
python select_topn_from_sorted.py
```

#### 进行重排序训练

```bash
# 需要修改参数
# webq
cd Model/rerank/webq
python main_listwise_compq.py
# compq
cd Model/rerank/compq
python main_listwise_webq.py
```
