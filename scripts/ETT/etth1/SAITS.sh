export CUDA_VISIBLE_DEVICES=2

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

feature_num=7

python -u run_models.py \
  --dataset_path ./generated_datasets/ETTh1_mr0125/ \
  --model_name SAITS \
  --feature_num $feature_num \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/ETTh1_mr025/ \
  --model_name SAITS \
  --feature_num $feature_num \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/ETTh1_mr0375/ \
  --model_name SAITS \
  --feature_num $feature_num \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/ETTh1_mr05/ \
  --model_name SAITS \
  --feature_num $feature_num \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \
