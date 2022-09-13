## 重排序执行流程

### 获取训练集和验证集所有候选

```bash
cd src/build_model_data/
# 注意修改输入参数
python build_all_train_dev.py
```

### 基于训练好的第一阶段排序模型对所有候选进行打分

```bash
cd src/rerank/predict_cands
# 注意修改输入参数
python predict_train.py
python predict_dev.py
python predict_test.py
```

### 根据排序得分获取有序的候选查询图

```bash
cd src/rerank/predict_cands
python get_sorted_cands.py
```

### 根据有序的候选查询图构建重排序数据

```
cd src/rerank/build_data
python select_1_n.py
```

### 可进行重排序训练
