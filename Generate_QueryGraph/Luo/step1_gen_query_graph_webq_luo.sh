#!/bin/bash
# -*- coding: utf-8 -*-

export CUDA_VISIBLE_DEVICES=6
PY='python -u -m';  TASK=smart_candgen
DATA_NAME=WebQ
DS_NAME=20201202_entity_time_type_ordinal
OUTPUT_DIR=../../runnings/candgen_${DATA_NAME}/${DS_NAME}

mkdir -p ${OUTPUT_DIR}/logs/${DATA_NAME}
cp step1_gen_query_graph.sh ${OUTPUT_DIR}/exec.sh

# for var in {0..100..1}
# 00表示00-99第一组，01表示100-200第二组
# WebQ总共有5810个问题，所以需要0--58个
for var in $(seq 0 58)
# for var in $(seq 7 7)
do
    echo ${var}
    pad_var=$(printf "%02d" ${var})
    $PY $TASK \
        --group_id ${var} \
        --output_dir ${OUTPUT_DIR} \
        --data_name ${DATA_NAME} \
        > ${OUTPUT_DIR}/logs/${DATA_NAME}/recall_from_entity-${DATA_NAME}-${var}.log &
    sleep 3
done
