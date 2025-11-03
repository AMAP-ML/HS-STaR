#!/bin/bash
set -e
set -u
set -x

export PYTHONPATH=$(pwd)
iteration=$1
model_name=$2
cold_start_dataset=$3
method_name=$4
dataset=$5
prompt_type=$6
dataset_split=${7:-"middle"}
eval_num=${8:-3}
data_index_num=8
total_num=8
dataset_length=$(wc -l < "data/train/${dataset}_iteration_${iteration}.jsonl")
echo "###########################  SYNTH OURS DPO ###########################"

dataset_type="${dataset}_full"
seq 0 7 | xargs -P 8 -I{} bash -c "CUDA_VISIBLE_DEVICES={} bash scripts/synth_iteration_split.sh ${iteration} ${model_name} ${prompt_type} ${dataset} ${dataset_type} ${cold_start_dataset} ${method_name} {} ${data_index_num} ${eval_num}"
python scripts/merge_data.py --model_name ${model_name} --cold_start_dataset ${cold_start_dataset} --dataset ${dataset} --dataset_type ${dataset_type} --prompt_type ${prompt_type} --real_sampling_num ${eval_num} --iteration ${iteration} --data_index_num ${data_index_num} --method_name ${method_name}
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 skywork/eva_reward_sky_ddp.py \
    --data_file "data/res/${model_name}/iteration-${iteration}/synthetic/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}/${dataset_type}/synth_${dataset}_seed_random_n${eval_num}.jsonl" \
    --output_file "data/res/${model_name}/iteration-${iteration}/synthetic/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}/${dataset_type}/synth_${dataset}_seed_random_n${eval_num}_reward_dict_skywork.jsonl" \
    --batch_size 4

acc_extra_args=""
if [[ "$method_name" == *acc* ]]; then
    acc_extra_args="--eval_type acc"
elif [[ "$method_name" == *reward* ]]; then
    acc_extra_args="--eval_type prm"
fi
python scripts/classify_diff.py --model_name ${model_name} --dataset ${dataset} --iteration ${iteration} --cold_start_dataset ${cold_start_dataset} --prompt_type ${prompt_type} --method_name ${method_name} --eval_num ${eval_num} ${acc_extra_args}



dataset_type="${dataset}_${dataset_split}"    
dataset_type_lines=$(wc -l < "data/res/${model_name}/iteration-${iteration}/synthetic/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}/${dataset_type}/train.jsonl")
dataset_type_sampling_num=$(( ((total_num - eval_num) * dataset_length + dataset_type_lines - 1) / dataset_type_lines ))
echo "${dataset_type}_lines = ${dataset_type_lines}, ${dataset_type}_sampling_num = ${dataset_type_sampling_num}"
if [[ $dataset_type_sampling_num -eq 0 ]]; then
    echo "dataset_type_sampling_num is 0"
else
    echo "dataset_type_sampling_num is not 0"
    seq 0 7 | xargs -P 8 -I{} bash -c "CUDA_VISIBLE_DEVICES={} bash scripts/synth_iteration_split.sh ${iteration} ${model_name} ${prompt_type} ${dataset} ${dataset_type} ${cold_start_dataset} ${method_name} {} ${data_index_num} ${dataset_type_sampling_num}"
    python scripts/merge_data.py --model_name ${model_name} --cold_start_dataset ${cold_start_dataset} --dataset ${dataset} --dataset_type ${dataset_type} --prompt_type ${prompt_type} --real_sampling_num ${dataset_type_sampling_num} --iteration ${iteration} --data_index_num ${data_index_num} --method_name ${method_name}
    python scripts/post_process_residue.py --model_name ${model_name} --dataset ${dataset} --dataset_split ${dataset_split} --dataset_type ${dataset_type} --iteration ${iteration} --cold_start_dataset ${cold_start_dataset} --prompt_type ${prompt_type} --method_name ${method_name} --eval_num ${eval_num} --total_num ${total_num} --data_index_num ${data_index_num} --dataset_length ${dataset_length}
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 skywork/eva_reward_sky_ddp.py \
        --data_file "data/res/${model_name}/iteration-${iteration}/synthetic/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}/${dataset_type}/synth_${dataset}_seed_random_n${dataset_type_sampling_num}.jsonl" \
        --output_file "data/res/${model_name}/iteration-${iteration}/synthetic/${cold_start_dataset}/${dataset}/${prompt_type}/${method_name}/${dataset_type}/synth_${dataset}_seed_random_n${dataset_type_sampling_num}_reward_dict_skywork.jsonl" \
        --batch_size 4
fi
python scripts/filter_ours_dpo.py --model_name ${model_name} --dataset ${dataset} --dataset_split ${dataset_split} --dataset_type ${dataset_type} --iteration ${iteration} --cold_start_dataset ${cold_start_dataset} --prompt_type ${prompt_type} --method_name ${method_name} --eval_num ${eval_num} --re_sampling_num ${dataset_type_sampling_num}


