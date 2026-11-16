#!/usr/bin/env bash
# Source in an existing terminal from the repository root:
#   conda activate parttwin_sim && source scripts/activate_sim.sh
# Does NOT create a Conda environment and does NOT install packages.
if [[ "${CONDA_DEFAULT_ENV:-}" != "parttwin_sim" ]]; then
    echo "Activate the existing environment first: conda activate parttwin_sim" >&2
    return 1 2>/dev/null || exit 1
fi

# --- Python isolation -------------------------------------------------------
# ROS Jazzy (Python 3.12) and unrelated workspaces leak through PYTHONPATH and
# register pytest plugins (launch_testing) that break collection in Python 3.11.
unset PYTHONPATH
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"

# --- CUDA toolchain for the gsplat JIT extension -----------------------------
# Conda's CUDA 12.8 packages place headers/libs under targets/x86_64-linux.
# nvcc finds them via nvcc.profile, but the *host* C++ compiler used for
# gsplat's .cpp bindings does not; PyTorch only adds $CUDA_HOME/include.
# Without CPATH the build fails with "fatal error: cuda_runtime.h".
# These are environment variables (not compiler flags), so they do not change
# ninja command lines and do not invalidate the cached CUDA objects.
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
_pt_cuda_target="${CONDA_PREFIX}/targets/x86_64-linux"
if [[ -d "${_pt_cuda_target}/include" ]]; then
    case ":${CPATH:-}:" in
        *":${_pt_cuda_target}/include:"*) ;;
        *) export CPATH="${_pt_cuda_target}/include${CPATH:+:${CPATH}}" ;;
    esac
fi
unset _pt_cuda_target
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS="${MAX_JOBS:-2}"
# Conda CUDA toolkit ships GCC 14 wrappers compatible with CUDA 12.8. The
# existing gsplat build cache (~/.cache/torch_extensions/py311_cu128/gsplat_cuda)
# was compiled with these; switching compilers forces a full rebuild.
[[ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc" ]] && export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
[[ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]] && export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
# Do not override TORCH_EXTENSIONS_DIR: keep the persistent gsplat cache.

# --- Repository import path ------------------------------------------------
# The project is a src/ layout package installed editable (`make setup`).
# Fallback for a fresh checkout without the editable install:
_pt_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! python -c "import parttwin1d" >/dev/null 2>&1; then
    export PYTHONPATH="${_pt_root}/src"
fi
unset _pt_root

echo "parttwin_sim: CUDA_HOME=${CUDA_HOME} CPATH=${CPATH} (persistent gsplat extension cache)"
