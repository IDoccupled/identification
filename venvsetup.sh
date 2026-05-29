#!/bin/bash

mkdir -p ~/venv/venv_identification

python3 -m venv ~/venv/venv_identification --system-site-packages --symlinks

source ~/venv/venv_identification/bin/activate

script_dir="$(cd "$(dirname "$0")" && pwd)"

python3 -m pip install -r "$script_dir/requirements.txt"

echo "venv_identification install done. Do <source ~/venv/venv_identification/bin/activate>"