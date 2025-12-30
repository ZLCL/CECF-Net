export CUDA_VISIBLE_DEVICES=3

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

feature_num=321

python -u run_models.py \
  --dataset_path ./generated_datasets/electricity_mr0125/ \
  --model_name Transformer \
  --d_inner 2048 \
  --d_model 512 \
  --n_head 8 \
  --d_k 64 \
  --feature_num $feature_num \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/electricity_mr025/ \
  --model_name Transformer \
  --d_inner 2048 \
  --d_model 512 \
  --n_head 8 \
  --d_k 64 \
  --feature_num $feature_num \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/electricity_mr0375/ \
  --model_name Transformer \
  --d_inner 2048 \
  --d_model 512 \
  --n_head 8 \
  --d_k 64 \
  --feature_num $feature_num \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/electricity_mr05/ \
  --model_name Transformer \
  --d_inner 2048 \
  --d_model 512 \
  --n_head 8 \
  --d_k 64 \
  --feature_num $feature_num \
  --device cuda \
  --lr 0.001 \
