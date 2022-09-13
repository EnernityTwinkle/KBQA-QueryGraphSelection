BASE_DIR=$(cd $(dirname $0);cd ..;cd ..; pwd)
echo $BASE_DIR

nohup python -u train_listwise.py \
  --config_file $BASE_DIR'/config/CCKS2021_stage1.yaml' \
> log/CCKS2021_stage1.log&