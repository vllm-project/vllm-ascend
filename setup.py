from setuptools import setup

setup(name='vllm_ascend_plugin',
      version='0.1',
      packages=['vllm_ascend_plugin'],
      entry_points={
          'vllm.general_plugins':
          ["ascend_plugin = vllm_ascend_plugin:register"]
      })
