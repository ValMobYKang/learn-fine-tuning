#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate

if [[  $(which python3) != *"$(pwd)"* ]]; then
    echo "Error: current $(which python3) and target $(pwd) does not match."
    echo "Exiting."
    exit;
fi

pip install -r ./requirements.txt