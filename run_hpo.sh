#!/bin/sh

# DATA_ROOT=./data


echo "=== running hyper parameter optimisation for adversarial contrastive learning ==="
echo "---"

python3 run_hpo.py

echo "=== hpo complete ==="

