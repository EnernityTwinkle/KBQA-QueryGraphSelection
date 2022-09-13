BASE_DIR=$(cd $(dirname $0);cd ..;cd ..;cd ..; pwd)
echo $BASE_DIR

nohup python -u train_rerank.py \
  --config_file $BASE_DIR'/config/CCKS2021_stage2.yaml' \
> ccks2021_stage2.log&