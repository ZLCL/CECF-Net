export CUDA_VISIBLE_DEVICES=2

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

batch_size=16

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Ali_mr0375_ft02/ \
  --model_name Transformer \
  --d_inner 2048 \
  --d_model 512 \
  --n_head 8 \
  --d_k 64 \
  --feature_num 25 \
  --batch_size $batch_size \
  --device cuda \
  --lr 0.001 \


python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Arou_mr0375_ft02/ \
  --model_name Transformer \
  --d_inner 2048 \
  --d_model 512 \
  --n_head 8 \
  --d_k 64 \
  --feature_num 11 \
  --batch_size $batch_size \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Mt.Everest_mr0375_ft02/ \
  --model_name Transformer \
  --d_inner 2048 \
  --d_model 512 \
  --n_head 8 \
  --d_k 64 \
  --feature_num 36 \
  --batch_size $batch_size \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Muztag_mr0375_ft02/ \
  --model_name Transformer \
  --d_inner 2048 \
  --d_model 512 \
  --n_head 8 \
  --d_k 64 \
  --feature_num 20 \
  --batch_size $batch_size \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-NamCo_mr0375_ft02/ \
  --model_name Transformer \
  --d_inner 2048 \
  --d_model 512 \
  --n_head 8 \
  --d_k 64 \
  --feature_num 35 \
  --batch_size $batch_size \
  --device cuda \
  --lr 0.001 \
