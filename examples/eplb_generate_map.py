import argparse
import json
import random

import numpy as np


def add_unique_number_with_retry(existing_numbers,
                                 valid_range,
                                 max_attempts=100):
    '''
    generate an unique number not in existing_numbers
    Args:
        existing_numbers:
        valid_range:
        max_attempts:

    Returns:

    '''
    existing_set = set(existing_numbers)
    min_val, max_val = valid_range
    for _ in range(max_attempts):
        candidate = random.randint(min_val, max_val)
        if candidate not in existing_set:
            return candidate

    raise ValueError('No unique number found')


def split_and_insert(n, k, m):
    '''
    n: expert num
    k: card num
    m: redundant expert num, make sure m%k==0
    '''

    A = np.arange(n)

    groups = np.array_split(A, k)

    for j in range(m // k):
        for i in range(k):
            candidate = add_unique_number_with_retry(groups[i], (0, n - 1))
            groups[i] = np.append(groups[i], candidate)
    return np.concatenate(groups)


def random_generation(n_layer=58,
                      n_expert=256,
                      start_layer_idx=0,
                      device_count=128,
                      n_redundant=128,
                      output_name=""):
    expert_data = {}
    expert_data["moe_layer_count"] = n_layer
    layer_list = []
    for i in range(n_layer):
        layer = {"layer_id": start_layer_idx + i, "device_count": device_count}
        random_placement = split_and_insert(n_expert, device_count,
                                            n_redundant)
        device_list = []
        step = random_placement.shape[0] // device_count
        for j in range(device_count):
            device = {}
            device["device_id"] = j
            device["device_expert"] = random_placement[j * step:(j + 1) *
                                                       step].tolist()
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)

    expert_data["layer_list"] = layer_list
    json_file_path = output_name

    with open(json_file_path, "w") as f:
        json.dump(expert_data, f, indent=4)

    print(f"JSON file generated: {json_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "python generate_map.py --n_layers 2 --n_experts 256 --card_num 8 --n_redundant 8 --output expert_map.json"
    )
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument("--n_experts", type=int, required=True)
    parser.add_argument("--card_num", type=int, required=True)
    parser.add_argument("--n_redundant", type=int, default=0)
    parser.add_argument("--output", type=str, default="expert_map.json")
    args = parser.parse_args()

    n_layers = args.n_layers
    n_experts = args.n_experts
    card_num = args.card_num
    n_redundant = args.n_redundant
    output = args.output

    random_generation(n_layers, n_experts, 0, card_num, n_redundant, output)
