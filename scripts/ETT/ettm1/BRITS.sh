export CUDA_VISIBLE_DEVICES=0

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

feature_num=7
batch_size=128

python -u run_models.py \
  --result_saving_base_dir $result_saving_base_dir \
  --dataset_path ./generated_datasets/ETTm1_mr0125/ \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --batch_size $batch_size\
  --seq_len 192 \
  --feature_num $feature_num \
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --result_saving_base_dir $result_saving_base_dir \
  --dataset_path ./generated_datasets/ETTm1_mr025/ \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size\
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --result_saving_base_dir $result_saving_base_dir \
  --dataset_path ./generated_datasets/ETTm1_mr0375/ \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size\
  --MIT False \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --result_saving_base_dir $result_saving_base_dir \
  --dataset_path ./generated_datasets/ETTm1_mr05/ \
  --model_name BRITS \
  --rnn_hidden_size 512 \
  --seq_len 192 \
  --feature_num $feature_num \
  --batch_size $batch_size\
  --MIT False \
  --device cuda \
  --lr 0.001 \

