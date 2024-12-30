from setuptools import setup

setup(name='vllm_ascend_plugin',
      version='0.1',
      packages=['vllm_ascend_plugin'],
      entry_points={
          'vllm.platform_plugins':
          ["ascend_plugin = vllm_ascend_plugin:register"]
      })
