import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from ascendc_extension import AscendCExtension
CURRENT_DIR = os.path.dirname(__file__)

package_name = 'elastic_ipc_utils'
setup(
    name=package_name,
    ext_modules=[
        AscendCExtension(
            name=package_name,
            sources=['../../csrc/elastic_ipc_utils/ipc_utils.cpp'],
            extra_library_dirs=[
                '/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/torch_npu/lib/'
            ],
            extra_libraries=[
                "torch_npu"
            ]
            
            ),
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)