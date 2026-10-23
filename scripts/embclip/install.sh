#!/usr/bin/env bash
# Independent runtime; the render stage is deliberately small for GPU preflight.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
stage=${1:-render}
[[ "$stage" == render || "$stage" == full ]] || { echo 'Usage: install.sh render|full' >&2; exit 2; }
prefix="$root/.envs/embclip-robothor"
cache="$root/assets/embclip-robothor"
mkdir -p "$cache/conda-pkgs" "$cache/pip-cache" "$root/logs/embclip-robothor-setup"
export CONDA_PKGS_DIRS="$cache/conda-pkgs"
export PIP_CACHE_DIR="$cache/pip-cache"
if [[ ! -x "$prefix/bin/python" ]]; then
  conda create --prefix "$prefix" --override-channels -c conda-forge python=3.8 pip -y
fi
"$prefix/bin/python" -m pip install 'pip<25' 'setuptools<66' wheel
"$prefix/bin/python" -m pip install 'ai2thor==4.3.0' 'numpy==1.23.5' 'Pillow==8.4.0'
if [[ "$stage" == full ]]; then
  "$prefix/bin/python" -m pip install 'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
  "$prefix/bin/python" -m pip install 'gym==0.17.3' 'networkx==2.5' 'imageio==2.9.0' 'imageio-ffmpeg==0.4.3' 'moviepy==1.0.3' 'cloudpickle==1.6.0' 'compress-pickle==1.2.0' 'tensorboardX==2.1' 'protobuf==3.20.3' 'scipy==1.10.1' 'matplotlib==3.5.3' 'opencv-python-headless==4.8.1.78' setproctitle attrs 'numba==0.58.1' colour packaging 'numpy-quaternion==2023.0.4' pyquaternion python-xlib 'ftfy==6.1.3' regex tqdm 'typing_extensions<4.14' 'importlib_metadata<8.6'
  "$prefix/bin/python" -m pip install --no-deps 'git+https://github.com/openai/CLIP.git@e184f608c5d5e58165682f7c332c3a8b4c1545f2'
fi
"$prefix/bin/python" -m pip freeze > "$root/logs/embclip-robothor-setup/pip-freeze-$stage.txt"
"$prefix/bin/python" -c 'import ai2thor, ai2thor.build; print("ai2thor", ai2thor.__version__, "build", ai2thor.build.COMMIT_ID)'
