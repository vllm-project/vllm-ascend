import multiprocessing
import os
import sys

dp_size = 32
dp_size_local = 16
dp_rank_start = 0
dp_ip = "your_dp_ip_here"
dp_port = "your_dp_port_here"
engine_port = 9000
template_path = "./run_dp_template.sh"
if not os.path.exists(template_path):
  print(f"Template file {template_path} does not exist.")
  sys.exit(1)

def run_command(dp_rank_local, dp_rank, engine_port_):
  command = f"bash ./run_dp_template.sh {dp_size} {dp_ip} {dp_port} {dp_rank_local} {dp_rank} {engine_port_} {dp_size_local}"
  os.system(command)

processes = []
for i in range(dp_size_local):
  dp_rank = dp_rank_start + i
  dp_rank_local = i
  engine_port_ = engine_port + i
  process = multiprocessing.Process(target=run_command, args=(dp_rank_local, dp_rank, engine_port_))
  processes.append(process)
  process.start()

for process in processes:
  process.join()
