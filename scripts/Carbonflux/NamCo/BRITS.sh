export CUDA_VISIBLE_DEVICES=2

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

feature_num=35
batch_size=16

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-NamCo_mr0125_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-NamCo_mr025_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-NamCo_mr0375_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-NamCo_mr05_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \