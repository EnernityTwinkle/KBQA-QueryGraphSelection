# Two-stage Query Graph Selection for Knowledge Base Question Answering

代码 文档还在整理中, 目前上传的还是中间版本。后续会新增EN-readme

更多细节可参考 __Yonghui Jia, Chuanyuan Tan, Yuehe Chen, Muhua Zhu, Pingfu Chao, Wenliang Chen. Two-stage Query Graph Selection for Knowledge Base Question Answering. NLPCC 2022.__

中文数据集上的实验参考ckbqa/README.md 展开(待上传)

## setups[todo]

代码在下述环境中测试

## 💾 数据获取

- [CCKS2019-KBQA](https://www.biendata.xyz/competition/ccks_2019_6/) 源自CCKS2019评测任务: 中文知识图谱问答

- [CCKS2021-KBQA](https://www.biendata.xyz/competition/ccks_2021_ckbqa/) 源自CCKS2021评测任务: 生活服务知识图谱问答评测

- [WebQuestions](https://nlp.stanford.edu/software/sempre/) 源自论文[Semantic Parsing on Freebase from Question-Answer Pairs](https://aclanthology.org/D13-1160/)

- [ComplexQuestions](https://github.com/JunweiBao/MulCQA/tree/ComplexQuestions) 源自论文[Constraint-Based Question Answering with Knowledge Graph](https://aclanthology.org/C16-1236.pdf)

## 🚀 快速复现实验结果

### 1️⃣ 查询图生成

此步骤非必要, 我们提供各数据集的生成结果，供排序使用

- 生成WebQ数据集对应的候选查询图

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
python build_listwise_data_with_answer.py
```

- CompQ

```bash
cd Build_Data/CompQ/
python build_prerank_data.py
```

### 3️⃣ stage1 排序【重新训练中】

```bash
cd Model/Listwise

# CompQ
# todo 或许 后续调整传参方式
nohup python main_bert_listwise_comp.py > 0908_gpu7_stage1_comp_neg40.log&
# 还没验证
python main_bert_listwise_webq.py
```

### 4️⃣ 构建stage2 排序的输入数据【todo】

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

### 5️⃣ 重排序[todo]

#### 基于stage1模型对所有候选打分

``` bash
cd Model/prerank/pairwise/webq
python predict_dev_data_webq.py     根据训练好的排序模型计算验证集候选的得分
python predict_train_data_webq.py   根据训练好的排序模型计算训练集候选的得分
```

#### 根据排序得分获取有序的候选查询图[todo]

#### 根据有序的候选查询图构建重排序数据[todo]

#### 进行重排序训练[todo]

```bash
cd Model/rerank/webq

cd Model/rerank/compq
# 运行
```