# 获取训练集、验证集和测试集的所有候选

```
cd WebQ
python build_listwise_data_with_answer.py

cd CompQ
python build_prerank_data.py
```

# 构建重排序数据
根据训练好的预排序模型产生的得分，选出top-n
```
cd WebQ
python get_sorted_cand_from_prerank_score.py
python selet_topn_from_sorted.py
python select_1_n.py
```

```
cd CompQ
python get_sorted_cand_from_prerank.py
python selet_topn_from_sorted.py
python select_1_n.py
```
