set -ex
# ===== 参数配置 =====
lr=$1
method_name=$2
model_name=$3
cold_start_dataset=$4
train_data_name=$5
prompt_type=${6:-"qwen25-step-cot"}
eval_num=${7:-3}
dataset=${8:-"math"}

synth_file="scripts/synth_iteration_dpo.sh"
train_file="scripts/train_iteration_dpo.sh"
full_method_name=${method_name}


# ====== Training ======
train_iteration() {
    local synth_iter=$1
    local train_from_iter=$2
    local train_to_iter=$3
    bash "$synth_file" "$synth_iter" "$model_name" "$cold_start_dataset" "$full_method_name" "$dataset" "$prompt_type" "$train_data_name" "$eval_num"
    bash "$train_file" "$train_from_iter" "$train_to_iter" "$model_name" "$cold_start_dataset" "$full_method_name" "$dataset" "$prompt_type" "$train_data_name" "$lr"
}

# ====== Evaluation ======
eval_iteration() {
    local iter=$1
    CUDA_VISIBLE_DEVICES="0,1,2,3" bash scripts/eval_iteration_easy.sh "$iter" "$cold_start_dataset" "$model_name" \
        "$prompt_type" "$dataset" "$full_method_name" &
    CUDA_VISIBLE_DEVICES="4,5,6,7" bash scripts/eval_iteration_hard.sh "$iter" "$cold_start_dataset" "$model_name" \
        "$prompt_type" "$dataset" "$full_method_name" &
    wait
}

# ====== Iterative Training ======
for i in {0..2}; do
    next=$((i+1))
    train_iteration "$i" "$i" "$next"
done

eval_iteration 1
eval_iteration 2
eval_iteration 3

python scripts/eval_results.py --log_dir "data_eval/${model_name}/iteration-1/${cold_start_dataset}/${dataset}/${prompt_type}/${full_method_name}"
python scripts/eval_results.py --log_dir "data_eval/${model_name}/iteration-2/${cold_start_dataset}/${dataset}/${prompt_type}/${full_method_name}"
python scripts/eval_results.py --log_dir "data_eval/${model_name}/iteration-3/${cold_start_dataset}/${dataset}/${prompt_type}/${full_method_name}"

echo "All done."