# use eval for not iteration 0


iteration=$1
cold_start_dataset=$2
model_name=$3
prompt_type=$4
dataset=$5
method_name=$6
######################################################################################################
set -ex
export PYTHONPATH=$(pwd)
core_name="${model_name}/iteration-${iteration}/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}"

if [ "$iteration" -eq 0 ]; then
    if [ "$cold_start_dataset" = "none" ]; then
        model_path="/mnt/workspace/common/models/${model_name}"
        output_path="data_eval/${model_name}/iteration-${iteration}/${cold_start_dataset}/"
    else
        if [ "$model_name" = "Qwen2.5-7B" ]; then
            model_path="outputs/${model_name}/iteration-${iteration}/${cold_start_dataset}/sft_1e-6"
            output_path="data_eval/${model_name}/iteration-${iteration}/${cold_start_dataset}/sft_1e-6"
        elif [ "$model_name" = "Qwen2.5-3B" ]; then
            model_path="outputs/${model_name}/iteration-${iteration}/${cold_start_dataset}/sft_5e-6"
            output_path="data_eval/${model_name}/iteration-${iteration}/${cold_start_dataset}/sft_5e-6"
        else
            echo "Error: Unsupported model name '$model_name'" >&2
            exit 1
        fi
        
    fi    
else
    model_path="outputs/${core_name}"
    output_path="data_eval/${core_name}"
fi

# echo "=== START gsm8k,math500,minerva_math,olympiadbench,college_math ==="
# bash simplerl_math_eval/sh/eval.sh ${prompt_type} ${model_path} ${output_path} 0 2048 0.95 "gsm8k,math500,minerva_math,olympiadbench,college_math" true 1
echo "=== START aime24,amc23 ==="
bash simplerl_math_eval/sh/eval.sh ${prompt_type} ${model_path} ${output_path} 0.6 2048 0.95 "aime24,amc23" true 32
echo ${output_path}
python scripts/eval_results.py --log_dir ${output_path}