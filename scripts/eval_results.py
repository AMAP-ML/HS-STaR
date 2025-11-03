import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer
from dart_math.utils import load_jsonl
from tqdm import tqdm
import argparse

def load_data(log_dir, dataset):
    """统一加载数据的辅助函数"""
    for file_name in os.listdir(os.path.join(log_dir, dataset)):
        if 'metrics' not in file_name:
            # print(os.path.join(log_dir, dataset, file_name))
            data = load_jsonl(os.path.join(log_dir, dataset, file_name))
            return data
    raise FileNotFoundError(f"No data file found for {dataset}")


def load_metrics(log_dir, dataset):
    """统一加载指标的辅助函数"""
    for file_name in os.listdir(os.path.join(log_dir, dataset)):
        if 'metrics' in file_name:
            with open(os.path.join(log_dir, dataset, file_name), "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"No metrics file found for {dataset}")

def calculate_averages(data):
    """计算加权平均值（考虑不同数据集的样本量）"""
    total = sum(val * metrics["num_samples"] for val, metrics in data)
    total_samples = sum(metrics["num_samples"] for _, metrics in data)
    return total / total_samples if total_samples != 0 else 0

if __name__ == "__main__":
    # 配置路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="data_eval/Qwen2.5-7B/iteration-3/math_filtered/numina/qwen25-step-cot-enter/eval_4_sampling_4/sft_first_simple_k1_5e-6_1_128_dpo_middle_k1_1e-6_1_128/sft_first_simple_k1_5e-6_1_128_dpo_middle_k1_1e-6_1_128-1")
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-7B")
    args = parser.parse_args()

    datasets = "gsm8k,math500,olympiadbench,minerva_math,amc23,college_math,aime24".split(",")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    results = defaultdict(dict)
    for dataset in tqdm(datasets):
        metrics = load_metrics(args.log_dir, dataset)
        dataset_data = load_data(args.log_dir, dataset)
        
        # print(metrics)
        # 第一轮计算（AIME/AMC用avg@k）
        if dataset in ["aime24", "amc23"]:
            results[dataset][f'avg@k'] = metrics[f"avg@k"]
            results[dataset][f'pass@k'] = metrics[f"pass@k"]["32"]
        else:
            results[dataset]['avg@k'] = metrics["acc"]
            results[dataset]['pass@k'] = metrics["acc"]
        

        lengths = []
        for item in dataset_data:
            for code in item["code"]:
                lengths.append(
                    len(tokenizer(code).input_ids)
                )
        # results[dataset]['max_length'] = max([length for length in lengths if length < 4000])
        results[dataset]['length'] = sum(lengths) / len(lengths)
            
    # print(results)
    dataset_acc_avg_results = f"dataset"
    dataset_acc_avg_pass_results = f""
    # Extract data
    avg_values = [results[dataset]['avg@k'] for dataset in datasets]
    pass_values = [results[dataset]['pass@k'] for dataset in datasets]
    avg_mean = sum(avg_values) / len(avg_values)
    pass_mean = sum(pass_values) / len(pass_values)

    # Print aligned table
    print(f"{'Dataset':<15} {'avg@k':<10} {'pass@k':<10}")  # 列标题
    for name, avg, pss in zip(datasets, avg_values, pass_values):
        print(f"{name:<15} {avg:<10.2f} {pss:<10.2f}")      # 数据集行
    print("-" * 36)                                         # 分隔线
    print(f"{'Average':<15} {avg_mean:<10.2f} {pass_mean:<10.2f}")  # 平均值行


    # Print length results
    print("\n\n")
    print (f"{'Dataset':<15} {'length':<10}")  # 列标题
    for name, length in zip(datasets, [results[dataset]['length'] for dataset in datasets]):
        print(f"{name:<15} {length:<10.2f}")

    # print("\n\n")
    # print (f"{'Dataset':<15} {'length':<10} {'max_length':<10}")  # 列标题
    # for name, length, max_length in zip(datasets, [results[dataset]['length'] for dataset in datasets], [results[dataset]['max_length'] for dataset in datasets]):
    #     print(f"{name:<15} {length:<10.2f} {max_length:<10.2f}")

