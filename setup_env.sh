#!/usr/bin/env bash
# Source this script from an activated Conda parttwin_sim shell.
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo 'ERROR: activate parttwin_sim first: conda activate parttwin_sim' >&2
  return 1 2>/dev/null || exit 1
fi
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=2
# If present, choose CUDA 12.8-compatible Conda host C/C++ compilers.
if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
fi
if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" ]; then
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
fi
# ROS Jazzy Python 3.12 site-packages and pytest entry points are not needed.
unset PYTHONPATH
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
# Do not override TORCH_EXTENSIONS_DIR; retain the cache path used by gsplat.
echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc=$(command -v nvcc)"
nvcc --version | grep -E 'release [0-9]+\.[0-9]+' || true
