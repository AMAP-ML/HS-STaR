from dart_math.utils import load_jsonl, save_jsonl, ensure_dir_exists
import argparse
import random
import os
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cold_start_dataset", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--real_sampling_num", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--data_index_num", type=int, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    args = parser.parse_args()
    data_folder = f"data/res/{args.model_name}/iteration-{args.iteration}/synthetic/{args.cold_start_dataset}/{args.dataset}/{args.prompt_type}/{args.method_name}/{args.dataset_type}"

    data_files = [ 
        os.path.join(data_folder, f"synth_{args.dataset}_seed_random_split{data_index}_n{args.real_sampling_num}.jsonl")
        for data_index in range(args.data_index_num)
    ]

    total_data = []

    for data_file in data_files:
        total_data.extend(load_jsonl(data_file))
    
    ensure_dir_exists(os.path.join(data_folder, f"synth_{args.dataset}_seed_random_n{args.real_sampling_num}.jsonl"))
    save_jsonl(total_data, os.path.join(data_folder, f"synth_{args.dataset}_seed_random_n{args.real_sampling_num}.jsonl"))
    
