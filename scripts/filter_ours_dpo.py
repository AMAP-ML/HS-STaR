import json
import os
import numpy as np
from collections import defaultdict
from dart_math.utils import load_json, load_jsonl, save_jsonl, ensure_dir_exists
from simplerl_math_eval.parser import parse_question
import argparse
from collections import defaultdict
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B")
    parser.add_argument("--dataset", type=str, default="numina")
    parser.add_argument("--dataset_split", type=str, default="middle")
    parser.add_argument("--dataset_type", type=str, default="numina_middle")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--cold_start_dataset", type=str, default='math_filtered')
    parser.add_argument("--prompt_type", type=str, default='qwen25-step-cot-enter')
    parser.add_argument("--method_name", type=str, default='ours_dpo_from_bl')
    parser.add_argument("--eval_num", type=int, default=3)
    parser.add_argument("--re_sampling_num", type=int, default=8)
    
    args = parser.parse_args()
    random.seed(1234)

    data_folder = f"data/res/{args.model_name}/iteration-{args.iteration}/synthetic/{args.cold_start_dataset}/{args.dataset}/{args.prompt_type}/{args.method_name}"
    data_dict = defaultdict(list)
    if args.eval_num == 0:
        for dataset_split_data_item in load_jsonl(os.path.join(data_folder, f"{args.dataset_type}", f'{args.dataset_split}.jsonl')):
            data_dict[dataset_split_data_item['query']].append(dataset_split_data_item)
        total_reward_data = load_jsonl(os.path.join(data_folder, f"{args.dataset_type}", f"synth_{args.dataset}_seed_random_n{args.re_sampling_num}_reward_dict_skywork.jsonl"))
    elif args.eval_num == 8:
        for dataset_split_data_item in load_jsonl(os.path.join(data_folder, f"{args.dataset}_full", f'{args.dataset_split}.jsonl')):
            data_dict[dataset_split_data_item['query']].append(dataset_split_data_item)
        total_reward_data = load_jsonl(os.path.join(data_folder, f"{args.dataset}_full", f"synth_{args.dataset}_seed_random_n{args.eval_num}_reward_dict_skywork.jsonl"))
    else:
        print(os.path.join(data_folder, f"{args.dataset}_full", f"synth_{args.dataset}_seed_random_n{args.eval_num}_reward_dict_skywork.jsonl"))
        print(os.path.join(data_folder, f"{args.dataset_type}", f"synth_{args.dataset}_seed_random_n{args.re_sampling_num}_reward_dict_skywork.jsonl"))
        for dataset_split_data_item in load_jsonl(os.path.join(data_folder, f"{args.dataset}_full", f'{args.dataset_split}.jsonl')) + load_jsonl(os.path.join(data_folder, f"{args.dataset_type}", f'{args.dataset_split}.jsonl')):
            data_dict[dataset_split_data_item['query']].append(dataset_split_data_item)
        total_reward_data = load_jsonl(os.path.join(data_folder, f"{args.dataset}_full", f"synth_{args.dataset}_seed_random_n{args.eval_num}_reward_dict_skywork.jsonl")) + \
                        load_jsonl(os.path.join(data_folder, f"{args.dataset_type}", f"synth_{args.dataset}_seed_random_n{args.re_sampling_num}_reward_dict_skywork.jsonl"))

    print("############################################# FILTER OURS DPO DATA #############################################")
    print("Total data length = ", len(data_dict))

    total_reward_data_dict = {}
    for item in total_reward_data:
        if item['query'] not in total_reward_data_dict:
            total_reward_data_dict[item['query']] = {}
        total_reward_data_dict[item['query']][item['resp']] = item['reward']

    construct_dpo_data = []
    for query in data_dict:
        chosen_resps = [data_item['resp'] for data_item in data_dict[query] if data_item['correct']]
        rejected_resps = [data_item['resp'] for data_item in data_dict[query] if not data_item['correct']]
        if len(chosen_resps) == 0 or len(rejected_resps) == 0:
            continue

        sorted_chosen = sorted(chosen_resps, key=lambda x: np.min(total_reward_data_dict[query][x]), reverse=True)
        sorted_rejected = sorted(rejected_resps, key=lambda x: np.min(total_reward_data_dict[query][x]), reverse=True)
        K = min(len(sorted_chosen), len(sorted_rejected))
        sorted_chosen = sorted_chosen[:K]
        sorted_rejected = sorted_rejected[:K]
        random.shuffle(sorted_chosen)
        random.shuffle(sorted_rejected)

        for chosen_resp, rejected_resp in zip(sorted_chosen,  sorted_rejected):
            construct_dpo_data.append({
                "prompt": query,
                "chosen": chosen_resp,
                "rejected": rejected_resp
            })

    random.shuffle(construct_dpo_data)
    ensure_dir_exists(os.path.join(data_folder, f"{args.dataset_split}", "train.jsonl"))
    save_jsonl(construct_dpo_data, os.path.join(data_folder, f"{args.dataset_split}", "train.jsonl"))
    print("data length = ", len(construct_dpo_data))