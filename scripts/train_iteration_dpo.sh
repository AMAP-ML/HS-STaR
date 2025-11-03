#! /bin/bash
# Default arguments
set -e
set -u
set -x

original_iteration=$1
iteration=$2
model_name=$3
cold_start_dataset=$4
method_name=$5
dataset=$6
prompt_type=$7
train_data_name=$8
lr=$9


export PYTHONPATH=$(pwd)
query_field="query"
resp_field="resp"
bs=256
n_grad_acc_steps=8
n_epochs=1
gpu_ids="0,1,2,3,4,5,6,7"

echo "###########################  TRAIN OURS DPO ###########################"
############################################ Need to check ############################################
# prompt_type="qwen25-step-cot-enter"                     
# dataset="numina"
# cold_start_dataset="none"
# model_name="Qwen2.5-Math-7B"
# cold_start_dataset="math_filtered"
# model_name="Qwen2.5-7B"
# method_name="ours_dpo"
# train_data_name="middle"
######################################################################################################
data_path="data/res/${model_name}/iteration-${original_iteration}/synthetic/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}/${train_data_name}"
if [ "$original_iteration" -eq 0 ]; then
    if [ "$cold_start_dataset" = "none" ]; then
        model_path="models/${model_name}"
    else
        if [ "$model_name" = "Qwen2.5-7B" ]; then
            model_path="outputs/${model_name}/iteration-${original_iteration}/${cold_start_dataset}/sft_1e-6"
        else
            echo "Error: Unsupported model name '$model_name'" >&2
            exit 1
        fi
    fi
else
    model_path="outputs/${model_name}/iteration-${original_iteration}/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}"
fi
output_dir="outputs/${model_name}/iteration-${iteration}/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}"


n_gpus=$(echo "${gpu_ids}" | awk -F, '{print NF}')
train_bs_per_gpu=$((bs / n_gpus))
compute_train_bs_per_gpu=$((train_bs_per_gpu / n_grad_acc_steps))

first_gpu_id=$(echo "${gpu_ids}" | cut -d',' -f1)
main_process_port=$((29500 + first_gpu_id)) # 29500 is the default port for DeepSpeed
# Plus the first GPU ID to avoid port conflicts when training multiple modes on the same machine

# Shared arguments
# deepspeed_config_file="cfgs/deepspeed/zero-stage1.conf"
deepspeed_config_file="cfgs/deepspeed/zero-stage2.conf"
# deepspeed_config_file="cfgs/deepspeed/zero-stage3.conf"
torch_compile_backend="inductor"

# NOTE: `${data_path}` is deliberately not quoted to pass possibly multiple values
# Launch training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes "${n_gpus}" \
    --num_cpu_threads_per_process $(($(nproc) / 2 / n_gpus)) \
    --gpu_ids "${gpu_ids}" \
    --same_network \
    --machine_rank 0 \
    --main_process_ip localhost \
    --main_process_port "${main_process_port}" \
    --use_deepspeed \
    --rdzv_backend static \
    --deepspeed_config_file "${deepspeed_config_file}" \
    --zero3_init_flag False \
    --dynamo_backend "${torch_compile_backend}" \
    pipeline/train.py \
    --data_path ${data_path} \
    --prompt_template ${prompt_type} \
    --model_name_or_path "${model_path}" \
    --model_max_length 2048 \
    --pack_len -1 \
    --shuffle_seed 3423 \
    --seed 3423 \
    --training_choice dpo \
    --per_device_train_batch_size "${compute_train_bs_per_gpu}" \
    --gradient_checkpointing True \
    --gradient_accumulation_steps "${n_grad_acc_steps}" \
    --num_train_epochs "${n_epochs}" \
    --logging_nan_inf_filter False \
    --save_strategy no \
    --save_only_model True \
    --learning_rate "${lr}" \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --bf16 True \
    --tf32 True \
    --logging_strategy steps \
    --logging_steps 1 \
    --deepspeed "${deepspeed_config_file}" \
    --torch_compile True \
    --torch_compile_backend "${torch_compile_backend}" \
    --lr_scheduler_type cosine \
    --output_dir "${output_dir}"
