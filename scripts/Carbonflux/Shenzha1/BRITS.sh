export CUDA_VISIBLE_DEVICES=0

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

feature_num=9
batch_size=32
#epoch=50

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Shenzha1_mr0125_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Shenzha1_mr025_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Shenzha1_mr0375_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/carbon-Shenzha1_mr05_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \