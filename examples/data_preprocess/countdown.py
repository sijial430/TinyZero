"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import random

def gen_dataset(
    num_samples: int,
    num_operands: int = 4,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    operations: List[str] = ['+', '-', '*', '/'],
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset for countdown task.
    
    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility
        
    Returns:
        List of tuples containing (target, numbers, solution)
    """
    seed(seed_value)
    samples = []
    
    for _ in tqdm(range(num_samples)):
        # Generate random target
        target = randint(1, max_target)
        
        # Generate random numbers
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]
        
        # Remove duplicates
        numbers = list(set(numbers))
        
        # Ensure the number of unique numbers is correct
        while len(numbers) < num_operands:
            numbers.append(randint(min_number, max_number))
        
        if any(num < min_number or num > max_number for num in numbers):
            continue

        # Shuffle the numbers
        random.shuffle(numbers)
        
        samples.append({
            'target': target,
            'nums': numbers
        })
    
    return samples


def make_prefix(dp, template_type, add_power_operation=False, add_floor_operation=False):
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        if not add_power_operation and not add_floor_operation:
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
            User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. You can use parentheses to group numbers.Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
            Assistant: Let me solve this step by step.
            <think>"""
        elif add_power_operation and not add_floor_operation:
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
            User: Using the numbers {numbers}, create an equation that equals {target}. You can use more advanced arithmetic and exponential operations (+, -, *, /, **, math.sqrt()) and each number can only be used once. You can use parentheses to group numbers. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
            Assistant: Let me solve this step by step.
            <think>"""
        elif add_power_operation and add_floor_operation:
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
            User: Using the numbers {numbers}, create an equation that equals {target}. You can use more advanced arithmetic, exponential, and floor division operations (+, -, *, /, **, math.sqrt(), //, %) and each number can only be used once. You can use parentheses to group numbers. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
            Assistant: Let me solve this step by step.
            <think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. You can use parentheses to group numbers. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/countdown')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--add_power_operation', type=bool, default=True)
    parser.add_argument('--add_floor_operation', type=bool, default=True)
    args = parser.parse_args()

    data_source = 'countdown'
    
    # raw_dataset = gen_dataset(args.num_samples, args.num_operands, args.max_target, args.min_number, args.max_number, args.operations, args.seed)
    # ds5_1k = gen_dataset(num_samples=200000, num_operands=5, max_target=1000, min_number=1, max_number=100)
    # ds6_1k = gen_dataset(num_samples=200000, num_operands=6, max_target=1000, min_number=1, max_number=100)
    # raw_samples = ds5_1k + ds6_1k
    
    # ds3_10k = gen_dataset(num_samples=200000, num_operands=3, max_target=10000, min_number=1, max_number=1000)
    # ds4_10k = gen_dataset(num_samples=200000, num_operands=4, max_target=10000, min_number=1, max_number=1000)
    # raw_samples = ds3_10k + ds4_10k
    
    # ds3_100k = gen_dataset(num_samples=200000, num_operands=3, max_target=100000, min_number=1, max_number=10000)
    # ds4_100k = gen_dataset(num_samples=200000, num_operands=4, max_target=100000, min_number=1, max_number=10000)
    # raw_samples = ds3_100k + ds4_100k
    
    # random.shuffle(raw_samples)
    # raw_dataset = Dataset.from_list(raw_samples)
    # print(len(raw_dataset))
    # print(raw_dataset[0])
    
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4-Unique', split='train')
    
    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type, add_power_operation=args.add_power_operation, add_floor_operation=args.add_floor_operation)
            solution = {
                "target": example['target'],
                "numbers": example['nums']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'power_operation': args.add_power_operation,
                    'floor_operation': args.add_floor_operation,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
