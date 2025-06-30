import vllm_ascend.envs as envs
from vllm_ascend.envs import environment_variables

lines = [
    "# Environment Variables\n",
    "\n",
    "vllm-ascend uses the following environment variables to configure the system:\n",
    "\n",
    "| Name | Type | Default | Description |\n",
    "| ---- | ---- | ------- | ----------- |\n",
]

variable_names = dir(envs)
for name in variable_names:
    lines.append(environment_variables[name].doc)

with open("./docs/source/user_guide/environment_variables.md",
          "w",
          encoding="utf-8") as file:
    file.writelines(lines)
