import numpy as np
import json
import argparse


def split_and_insert(n, k, m):
    '''
    n: expert num
    k: card num
    m: redundant expert num, make sure m%k==0
    '''

    A = np.arange(n)

    B = np.random.choice(n, size=m, replace=False)

    groups = np.array_split(A, k)

    for j in range(m // k):
        for i in range(k):
            groups[i] = np.append(groups[i], B[i + j * k])
    return np.concatenate(groups)

def random_generation(n_layer=58, n_expert=256, start_layer_idx=0, device_count=320, n_redundant=32, output_name=""):
    expert_data = {}
    expert_data["moe_layer_count"] = n_layer
    layer_list = []
    for i in range(n_layer):
        layer = {"layer_id": start_layer_idx + i, "device_count": device_count}
        random_placement = split_and_insert(n_expert, device_count, n_redundant)
        print(f"random_placement={random_placement}")
        device_list = []
        step = (n_expert + n_redundant) // device_count
        print(f"step={step}")
        for j in range(device_count):
            device = {}
            device["device_id"] = j
            routed_expert = random_placement[j].tolist()
            while True:
                redundant_expert = np.random.choice(n_expert, size=1, replace=False)
                if redundant_expert[0].item() in routed_expert:
                    continue
                break
            device["device_expert"] = random_placement[j].tolist() + [redundant_expert[0].item()]
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)

    expert_data["layer_list"] = layer_list
    json_file_path = output_name

    with open(json_file_path, "w") as f:
        json.dump(expert_data, f, indent=4)

    print(f"JSON file generated: {json_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python generate_map.py --n_layers 2 --n_experts 256 --card_num 8 --n_redundant 8 --output expert_map.json")
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
