#! /bin/bash
# Default arguments
# sleep 10m

iteration=0
data_path="data/${YOUR_DOWNLOAD_STEP_TRAIN_DATA}/step_train.jsonl"
query_field="query"
resp_field="response"
model_name="Qwen2.5-7B"
prompt_type="qwen25-step-cot"
model_path="models/${model_name}"
lr="1e-6"
bs=256
n_grad_acc_steps=4
n_epochs=1
gpu_ids="0,1,2,3,4,5,6,7"
output_dir="outputs/${model_name}/iteration-0/"


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
    --zero3_init_flag True \
    --dynamo_backend "${torch_compile_backend}" \
    pipeline/train.py \
    --data_path ${data_path} \
    --query_field "${query_field}" \
    --resp_field "${resp_field}" \
    --prompt_template "${prompt_type}" \
    --tokenized_cache_home "" \
    --model_name_or_path "${model_path}" \
    --model_max_length 4096 \
    --pack_len -1 \
    --shuffle_seed 3407 \
    --seed 3407 \
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
