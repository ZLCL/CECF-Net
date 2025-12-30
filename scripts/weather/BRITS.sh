export CUDA_VISIBLE_DEVICES=2

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

feature_num=21

python -u run_models.py \
  --result_saving_base_dir $result_saving_base_dir \
  --dataset_path ./generated_datasets/weather_mr0125/ \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --result_saving_base_dir $result_saving_base_dir \
  --dataset_path ./generated_datasets/weather_mr025/ \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --result_saving_base_dir $result_saving_base_dir \
  --dataset_path ./generated_datasets/weather_mr0375/ \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --result_saving_base_dir $result_saving_base_dir \
  --dataset_path ./generated_datasets/weather_mr05/ \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --MIT False \
  --device cuda \
  --lr 0.001 \

