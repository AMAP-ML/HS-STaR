
import math
from dart_math.utils import load_jsonl, save_jsonl, ensure_dir_exists
import argparse
import random
import os
from collections import defaultdict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-Math-7B")
    parser.add_argument("--dataset", type=str, default="numina")
    parser.add_argument("--dataset_split", type=str, default="middle")
    parser.add_argument("--dataset_type", type=str, default="numina_middle")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--cold_start_dataset", type=str, default='none')
    parser.add_argument("--prompt_type", type=str, default='qwen25-step-cot-enter')
    parser.add_argument("--method_name", type=str, default='ours_sft_from_bl')
    parser.add_argument("--eval_num", type=int, default=3)
    parser.add_argument("--total_num", type=int, default=8)
    parser.add_argument("--data_index_num", type=int, default=8)
    parser.add_argument("--dataset_length", type=int, default=7500)
    args = parser.parse_args()
    random.seed(1234)

    data_folder = f'data/res/{args.model_name}/iteration-{args.iteration}/synthetic/{args.cold_start_dataset}/{args.dataset}/{args.prompt_type}/{args.method_name}'
    original_data_path = f"data/qwen/{args.dataset}_iteration/iteration-{args.iteration}/train.jsonl"                                                             # 原始训练数据

    dataset_split_data = load_jsonl(os.path.join(data_folder, args.dataset_type, 'train.jsonl'))
    sampling_budget = int((args.dataset_length * (args.total_num - args.eval_num) ) / len(dataset_split_data))
    real_budget = math.ceil((args.dataset_length * (args.total_num - args.eval_num)) / len(dataset_split_data))

    main_data = []
    residue_data = []
    reward_data_dict = []
    for data_index in range(args.data_index_num):
        current_data = load_jsonl(os.path.join(data_folder, args.dataset_type, f"synth_{args.dataset}_seed_random_split{data_index}_n{real_budget}.jsonl"))
        current_data_num = int(len(current_data) / real_budget)
        print(int(len(current_data) / real_budget) , len(current_data) / real_budget, len(current_data), real_budget)
        assert int(len(current_data) / real_budget) == len(current_data) / real_budget
        main_data.extend(current_data[:current_data_num * sampling_budget])
        residue_data.extend(current_data[current_data_num * sampling_budget:])

    residue_data = random.sample(residue_data, args.dataset_length * (args.total_num - args.eval_num) % len(dataset_split_data))
    total_data = main_data + residue_data
    save_jsonl(total_data, os.path.join(data_folder, args.dataset_type, f'{args.dataset_split}.jsonl'))