#!/usr/bin/env python3
"""Check active environment without triggering long CUDA rasterizer JIT compilation."""
import os
import sys
import importlib.metadata as meta

print('Python:', sys.version.split()[0])
print('Conda:', os.environ.get('CONDA_DEFAULT_ENV'))
print('CUDA_HOME:', os.environ.get('CUDA_HOME'))
for name in ('torch', 'mani_skill', 'sapien', 'gsplat', 'plyfile', 'scipy'):
    try:
        print(f'{name}: {meta.version(name)}')
    except meta.PackageNotFoundError:
        print(f'{name}: NOT INSTALLED')

import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('PyTorch CUDA:', torch.version.cuda)
from torch.utils.cpp_extension import CUDA_HOME as torch_cuda_home
print('PyTorch compiler CUDA_HOME:', torch_cuda_home)
# Import only the Python rendering function, so this does NOT compile CUDA kernels.
from gsplat.rendering import rasterization
print('gsplat Python rasterization API:', callable(rasterization))
