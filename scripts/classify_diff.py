import json
import os
import numpy as np
from collections import defaultdict
from dart_math.utils import load_json, load_jsonl, save_jsonl, ensure_dir_exists
from simplerl_math_eval.parser import parse_question
import argparse
from collections import defaultdict

def load_paths(paths):
    data = []
    for path in paths:
        data.extend(load_jsonl(path))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B")
    parser.add_argument("--dataset", type=str, default="numina")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--cold_start_dataset", type=str, default='math_filtered')
    parser.add_argument("--prompt_type", type=str, default='qwen25-step-cot-enter')
    parser.add_argument("--method_name", type=str, default='ours_sft')
    parser.add_argument("--eval_nums", type=int, default=3)
    parser.add_argument("--eval_type", type=str, default="prm,acc")
    parser.add_argument("--dataset_length", type=int, default=7500)
    parser.add_argument("--prm_model", type=str, default="Skywork-o1-Open-PRM-Qwen-2.5-7B")
    args = parser.parse_args()

    data_folder = f'data/res/{args.model_name}/iteration-{args.iteration}/synthetic/{args.cold_start_dataset}/{args.dataset}/{args.prompt_type}/{args.method_name}'
    original_data_path = f"data/qwen/{args.dataset}_iteration/iteration-{args.iteration}/train.jsonl"                                                             # 原始训练数据

    eval_data = load_jsonl(os.path.join(data_folder, f"{args.dataset}_full", f'synth_{args.dataset}_seed_random_n{args.eval_nums}.jsonl'))
    eval_reward_data = load_jsonl(os.path.join(data_folder, f"{args.dataset}_full", f'synth_{args.dataset}_seed_random_n{args.eval_nums}_reward_dict_skywork.jsonl'))


    ########## eval_data statics ##########    
    eval_data_dict = defaultdict(list)
    for eval_data_item in eval_data:
        eval_data_dict[eval_data_item['query']].append(eval_data_item)
    
    eval_reward_data_dict = {}
    for eval_reward_data_item in eval_reward_data:
        if eval_reward_data_item['query'] not in eval_reward_data_dict:
            eval_reward_data_dict[eval_reward_data_item['query']] = {eval_reward_data_item['resp']: eval_reward_data_item['reward']}
        else:
            eval_reward_data_dict[eval_reward_data_item['query']][eval_reward_data_item['resp']] = eval_reward_data_item['reward']


    ######################## filter ############################
    original_data_dict = {}
    for item in load_jsonl(original_data_path):
        original_data_dict[parse_question(item, 'numina')] = item 

    simple_original_data = []
    middle_original_data = []
    hard_original_data = []
    simple_data = []
    middle_data = []
    hard_data = []
    
    if args.prm_model == 'Skywork-o1-Open-PRM-Qwen-2.5-7B':
        if args.model_name == 'Qwen2.5-7B':
            left_threshold = 0.15
            right_threshold = 0.65
        else:
            raise NotImplementedError
    elif args.prm_model == 'Skywork-o1-Open-PRM-Qwen-2.5-1.5B':
        if args.model_name == 'Qwen2.5-7B':
            left_threshold = 0.25
            right_threshold = 0.5
        else:
            raise NotImplementedError
    elif args.prm_model == 'Qwen2.5-Math-PRM-7B':
        if args.model_name == 'Qwen2.5-7B':
            left_threshold = 0.55
            right_threshold = 0.99
        else:
            raise NotImplementedError
    
    for query in eval_data_dict.keys():
        eval_data_samples = eval_data_dict[query]
        eval_data_rewards = eval_reward_data_dict[query]
        reward_mins = [np.min(reward_item) for reward_item in eval_data_rewards.values()]
        reward_mins_mean = sum(reward_mins) / len(reward_mins)
        query_acc = sum([item['correct'] for item in eval_data_samples]) / len(eval_data_samples)
       
        if 'prm' in args.eval_type and 'acc' in args.eval_type:
            if query_acc == 1 and reward_mins_mean > right_threshold:
                simple_data.extend(eval_data_samples)
                simple_original_data.append(original_data_dict[query])
            elif query_acc == 0 and reward_mins_mean < left_threshold:
                hard_data.extend(eval_data_samples)
                hard_original_data.append(original_data_dict[query])
            else:
                middle_data.extend(eval_data_samples)
                middle_original_data.append(original_data_dict[query])
        else:
            raise NotImplementedError

    save_jsonl(simple_data, os.path.join(data_folder, f"{args.dataset}_full", f'simple.jsonl'))
    save_jsonl(middle_data, os.path.join(data_folder, f"{args.dataset}_full", f'middle.jsonl'))
    save_jsonl(hard_data, os.path.join(data_folder, f"{args.dataset}_full", f'hard.jsonl'))
    save_jsonl(simple_data + middle_data, os.path.join(data_folder, f"{args.dataset}_full", f'simple_middle.jsonl'))
    save_jsonl(middle_data + hard_data, os.path.join(data_folder, f"{args.dataset}_full", f'middle_hard.jsonl'))
    save_jsonl(simple_data + hard_data, os.path.join(data_folder, f"{args.dataset}_full", f'simple_hard.jsonl'))

    ensure_dir_exists(os.path.join(data_folder, f"{args.dataset}_simple", 'train.jsonl'))
    ensure_dir_exists(os.path.join(data_folder, f"{args.dataset}_middle", 'train.jsonl'))
    ensure_dir_exists(os.path.join(data_folder, f"{args.dataset}_hard", 'train.jsonl'))
    ensure_dir_exists(os.path.join(data_folder, f"{args.dataset}_simple_middle", 'train.jsonl'))
    ensure_dir_exists(os.path.join(data_folder, f"{args.dataset}_middle_hard", 'train.jsonl'))
    ensure_dir_exists(os.path.join(data_folder, f"{args.dataset}_simple_hard", 'train.jsonl'))

    save_jsonl(simple_original_data, os.path.join(data_folder, f"{args.dataset}_simple", 'train.jsonl'))
    save_jsonl(middle_original_data, os.path.join(data_folder, f"{args.dataset}_middle", 'train.jsonl'))
    save_jsonl(hard_original_data, os.path.join(data_folder, f"{args.dataset}_hard", 'train.jsonl'))    
    save_jsonl(simple_original_data + middle_original_data, os.path.join(data_folder, f"{args.dataset}_simple_middle", 'train.jsonl'))
    save_jsonl(middle_original_data + hard_original_data, os.path.join(data_folder, f"{args.dataset}_middle_hard", 'train.jsonl'))
    save_jsonl(simple_original_data + hard_original_data, os.path.join(data_folder, f"{args.dataset}_simple_hard", 'train.jsonl'))

    print("simple data: ", len(simple_data)/(len(eval_data)/args.dataset_length))
    print("middle data: ", len(middle_data)/(len(eval_data)/args.dataset_length))
    print("hard data: ", len(hard_data)/(len(eval_data)/args.dataset_length))
    print("need simple original data: ", len(simple_original_data))
    print("need middle original data: ", len(middle_original_data))
    print("need hard original data: ", len(hard_original_data))
    remain_budget = args.dataset_length * 8 - len(eval_data)
    remain_simple_each_query_budget = (args.dataset_length * 8 - len(eval_data))/len(simple_original_data)
    remain_middle_each_query_budget = (args.dataset_length * 8 - len(eval_data))/len(middle_original_data)
    remain_hard_each_query_budget = (args.dataset_length * 8 - len(eval_data))/len(hard_original_data)
    print("remain budget: ", remain_budget)
    print("remain simple each query budget: ", remain_simple_each_query_budget)
    print("remain middle each query budget: ", remain_middle_each_query_budget)
    print("remain hard each query budget: ", remain_hard_each_query_budget)