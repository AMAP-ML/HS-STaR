import os
import torch
import json
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from dart_math.utils import save_jsonl, ensure_dir_exists
from transformers import AutoTokenizer
from skywork.model_utils.prm_model import PRM_MODEL
from skywork.model_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards
from torch.cuda.amp import autocast


def init_distributed():
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), dist.get_rank()

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

class ResponseDataset(Dataset):
    def __init__(self, data_dict, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        for query in data_dict:
            for item in data_dict[query]:               
                self.samples.append({
                    'problem': query,
                    'response': item['resp'],
                    'correct': item['correct'],
                    'raw_response': item['resp'],
                    'query': query
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids, steps, reward_flags = prepare_input(
            problem=sample['problem'],
            response=sample['response'],
            tokenizer=self.tokenizer,
            step_token="\n\n"
        )
        return {
            'input_ids': input_ids,
            'reward_flags': reward_flags,
            'correct': sample['correct'],
            'resp': sample['raw_response'],
            'query': sample['query']
        }

def collate_fn(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    reward_flags = [item['reward_flags'] for item in batch]
    
    processed = prepare_batch_input_for_model(
        input_ids, reward_flags, tokenizer.pad_token_id
    )
    
    return {
        'input_ids': processed[0],
        'attention_mask': processed[1],
        'reward_flags': processed[2],
        'queries': [item['query'] for item in batch],
        'resps': [item['resp'] for item in batch],
        'corrects': [item['correct'] for item in batch]
    }

def process_with_dataloader(data_dict, model, tokenizer, batch_size, rank, world_size):
    # 创建自定义collate_fn的partial
    from functools import partial
    custom_collate = partial(collate_fn, tokenizer=tokenizer)

    dataset = ResponseDataset(data_dict, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset, shuffle=False, num_replicas=world_size, rank=rank),
        collate_fn=custom_collate,
        pin_memory=True
    )

    samples_rewards_list = []
    for batch in tqdm(dataloader, desc=f"Rank {rank}", disable=rank != 0):
        device = next(model.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        reward_flags = batch['reward_flags'].to(device)

        with torch.no_grad():
            _, _, rewards = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_probs=True
            )

        step_rewards = derive_step_rewards(rewards, reward_flags)

        for q, r, c, s in zip(batch['queries'], batch['resps'], batch['corrects'], step_rewards):
            samples_rewards_list.append({
                'query': q,
                'correct': c,
                'reward': s,
                'resp': r,
                'resp_split': r.split("\n\n")  # 保持原始分割方式
            })

    # 分布式结果收集
    all_samples = [None] * world_size
    dist.all_gather_object(all_samples, samples_rewards_list)

    if rank == 0:
        merged_samples = []
        for sublist in all_samples:
            if sublist:  # 处理可能为None的情况
                merged_samples.extend(sublist)
        return merged_samples
    return None

if __name__ == "__main__":
    local_rank, world_size, rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/res/Qwen2.5-7B/iteration-0/synthetic/math_filtered/numina/qwen25-step-cot-enter/sampling_4/synth_numina_first_seed_1.jsonl")
    parser.add_argument("--output_file", type=str, default="data/res/Qwen2.5-7B/iteration-0/synthetic/math_filtered/numina/qwen25-step-cot-enter/sampling_4/synth_numina_first_seed_1_pure_reward_dict_n_4.jsonl")
    parser.add_argument("--model_name", type=str, default="models/Skywork-o1-Open-PRM-Qwen-2.5-7B")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # 数据加载与广播
    if rank == 0:
        raw_data = load_jsonl(args.data_file)
        print(f"Rank 0 loaded {len(raw_data)} samples")
    else:
        raw_data = []
    
    data_list = [raw_data]
    dist.broadcast_object_list(data_list, src=0)
    raw_data = data_list[0]

    # 构建数据字典
    data_dict = {}
    for item in raw_data:
        data_dict.setdefault(item['query'], []).append(item)

    # 模型加载
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # model = PRM_MODEL.from_pretrained(
    #     args.model_name,
    # ).to(device).eval()
    # model = DDP(model, device_ids=[local_rank])
    model = PRM_MODEL.from_pretrained(
        args.model_name,
    ).to(device).eval()
    model = model.to(dtype=torch.bfloat16)  # 转换模型参数为BF16
    model = DDP(model, device_ids=[local_rank])

    # 数据处理
    result = process_with_dataloader(
        data_dict, model.module, tokenizer,
        args.batch_size, rank, world_size
    )

    # 结果保存
    if rank == 0:
        ensure_dir_exists(args.output_file)
        save_jsonl(result, args.output_file)
        print(f"Saved {len(result)} results to {args.output_file}")

    dist.destroy_process_group()