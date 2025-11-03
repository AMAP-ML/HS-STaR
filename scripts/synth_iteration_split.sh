#!/bin/bash
# set -ex

iteration=$1
model_name=$2
prompt_type=$3
dataset=$4
dataset_type=$5
cold_start_dataset=$6
method_name=$7
data_index=$8
data_index_num=$9
n_sampling=${10}

export PYTHONPATH=$(pwd)
task_id="${model_name}/iteration-${iteration}/synthetic/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}"

if [[ "$dataset_type" == "${dataset}_full" || "$dataset_type" == "${dataset}_full_${n_sampling}" ]]; then
    data_file="data/train/${dataset}_iteration_${iteration}.jsonl"
elif [ "$dataset_type" = "${dataset}_simple" ]; then
    data_file="data/res/${task_id}/${dataset}_simple/train.jsonl"
elif [ "$dataset_type" = "${dataset}_middle" ]; then
    data_file="data/res/${task_id}/${dataset}_middle/train.jsonl"
elif [ "$dataset_type" = "${dataset}_hard" ]; then
    data_file="data/res/${task_id}/${dataset}_hard/train.jsonl"
elif [ "$dataset_type" = "${dataset}_simple_middle" ]; then
    data_file="data/res/${task_id}/${dataset}_simple_middle/train.jsonl"
elif [ "$dataset_type" = "${dataset}_middle_hard" ]; then
    data_file="data/res/${task_id}/${dataset}_middle_hard/train.jsonl"
elif [ "$dataset_type" = "${dataset}_simple_hard" ]; then
    data_file="data/res/${task_id}/${dataset}_simple_hard/train.jsonl"
else
    echo "Unsupported dataset_type: $dataset_type" >&2
    exit 1
fi

total_lines=$(wc -l < "$data_file")
begin_index=$((total_lines * data_index / data_index_num))
end_index=$((total_lines * (data_index + 1) / data_index_num))

# 模型路径分支
if [ "$iteration" -eq 0 ]; then
    if [ "$cold_start_dataset" = "none" ]; then
        model_path="models/${model_name}"
    else
        if [ "$model_name" = "Qwen2.5-7B" ]; then
            model_path="outputs/${model_name}/iteration-0/"
        else
            echo "Error: Unsupported model name '$model_name'" >&2
            exit 1
        fi
    fi    
else
    model_path="outputs/${model_name}/iteration-${iteration}/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}"
fi



if [ -f "data/res/${task_id}/${dataset_type}/synth_${dataset}_seed_random_split${data_index}_n${n_sampling}.jsonl" ]; then
    rm "data/res/${task_id}/${dataset_type}/synth_${dataset}_seed_random_split${data_index}_n${n_sampling}.jsonl"
    echo "Deleted data/res/${task_id}/${dataset_type}/synth_${dataset}_seed_random_split${data_index}_n${n_sampling}.jsonl"
fi


mkdir -p "logs/${task_id}" "data/res/${task_id}/${dataset_type}"

python pipeline/gen.py \
    --gen_save_path "data/res/${task_id}/${dataset_type}/synth_${dataset}_seed_random_split${data_index}_n${n_sampling}.jsonl" \
    --model_name_or_path ${model_path} \
    --datasets ${data_file} \
    --max_new_toks 2048 --temperature 0.7 --top_p 0.95 \
    --prompt_template ${prompt_type} --n_shots 0 \
    --inf_seed -1 \
    --n_paths 1 --min_n_corrects 0 --max_n_trials ${n_sampling} \
    --eval_mode "numina" \
    --begin_index $begin_index --end_index $end_index \
    2>&1 | tee "logs/${task_id}/synth_${dataset_type}_index${data_index}_n${n_sampling}.log"
    # >"logs/${task_id}/synth_${dataset_type}_index${data_index}_n${n_sampling}.log" 2>&1