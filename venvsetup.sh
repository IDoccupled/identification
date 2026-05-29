#!/bin/bash

mkdir -p ~/venv/venv_indentification

python3 -m venv ~/venv/venv_indentification --system-site-packages --symlinks

source ~/venv/venv_indentification/bin/activate

python3 -m pip install -r requirements.txt

echo "venv_indentification install done. Do <source ~/venv/venv_indentification/bin/activate>"