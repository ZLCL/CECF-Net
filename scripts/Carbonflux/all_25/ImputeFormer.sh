export CUDA_VISIBLE_DEVICES=3

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

batch_size=16

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Ali_mr025_ft02 \
  --model_name ImputeFormer \
  --feature_num 25 \
  --batch_size $batch_size \
  --num_nodes 25 \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Arou_mr025_ft02 \
  --model_name ImputeFormer \
  --n_groups 2 \
  --n_group_inner_layers 1 \
  --feature_num 11 \
  --batch_size $batch_size \
  --num_nodes 11 \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Mt.Everest_mr025_ft02 \
  --model_name ImputeFormer \
  --n_groups 2 \
  --n_group_inner_layers 1 \
  --feature_num 36 \
  --batch_size $batch_size \
  --num_nodes 36 \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Muztag_mr025_ft02 \
  --model_name ImputeFormer \
  --n_groups 2 \
  --n_group_inner_layers 1 \
  --feature_num 20 \
  --batch_size $batch_size \
  --num_nodes 20 \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-NamCo_mr025_ft02 \
  --model_name ImputeFormer \
  --n_groups 2 \
  --n_group_inner_layers 1 \
  --feature_num 35 \
  --batch_size $batch_size \
  --num_nodes 35 \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \