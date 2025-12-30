export CUDA_VISIBLE_DEVICES=2

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

batch_size=16

python -u run_models.py \
  --feature_num 25 \
  --dataset_path ./generated_datasets/carbon-Ali_mr025_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --feature_num 11 \
  --dataset_path ./generated_datasets/carbon-Arou_mr025_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --feature_num 36 \
  --dataset_path ./generated_datasets/carbon-Mt.Everest_mr025_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --feature_num 20 \
  --dataset_path ./generated_datasets/carbon-Muztag_mr025_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --feature_num 35 \
  --dataset_path ./generated_datasets/carbon-NamCo_mr025_ft02 \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --batch_size $batch_size \
  --MIT False \
  --device cuda \
  --lr 0.001 \
