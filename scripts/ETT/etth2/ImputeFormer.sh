export CUDA_VISIBLE_DEVICES=2

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

feature_num=7
batch_size=32

python -u run_models.py \
  --dataset_path ./generated_datasets/ETTh2_mr0125 \
  --model_name ImputeFormer \
  --feature_num $feature_num \
  --num_nodes $feature_num \
  --batch_size $batch_size \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/ETTh2_mr025 \
  --model_name ImputeFormer \
  --n_groups 2 \
  --n_group_inner_layers 1 \
  --feature_num $feature_num \
  --num_nodes $feature_num \
  --batch_size $batch_size \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/ETTh2_mr0375 \
  --model_name ImputeFormer \
  --n_groups 2 \
  --n_group_inner_layers 1 \
  --feature_num $feature_num \
  --num_nodes $feature_num \
  --batch_size $batch_size \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/ETTh2_mr05 \
  --model_name ImputeFormer \
  --n_groups 2 \
  --n_group_inner_layers 1 \
  --feature_num $feature_num \
  --batch_size $batch_size \
  --num_nodes $feature_num \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \