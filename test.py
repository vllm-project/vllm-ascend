import torch
num_decodes = 2
query_start_loc_cpu = torch.tensor([0, 4, 8, 9])
decode_threshold = 4

actual_seq_lengths_q = ((torch.arange(num_decodes) + 1) * decode_threshold).tolist() + query_start_loc_cpu[1:].tolist()[
    num_decodes:
]


print(actual_seq_lengths_q)