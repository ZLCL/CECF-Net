export CUDA_VISIBLE_DEVICES=3

# create a dir to save logs and results
result_saving_base_dir='results'

if [ ! -d "$result_saving_base_dir" ]; then
  mkdir -p "$result_saving_base_dir"
fi

feature_num=321

# train a model
python -u run_models.py \
  --dataset_path ./generated_datasets/electricity_mr0125/ \
  --model_name TimesNet \
  --feature_num $feature_num \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/electricity_mr025/ \
  --model_name TimesNet \
  --feature_num $feature_num \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/electricity_mr0375/ \
  --model_name TimesNet \
  --feature_num $feature_num \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \

python -u run_models.py \
  --dataset_path ./generated_datasets/electricity_mr05/ \
  --model_name TimesNet \
  --feature_num $feature_num \
  --seq_len 192 \
  --device cuda \
  --lr 0.001 \


#python -run_models.py \
#  --save_imputations True \
#  --model_path $model_saving_dir/${test_model_name} \
#  --result_saving_path $test_results_saving_base_dir/${test_model_name}/${test_model_name} \
#  --testmode \