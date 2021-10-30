#!/bin/sh

# DATA_ROOT=./data


echo "=== installing dpendencies ==="
echo "---"

conda install -c anaconda pandas -y
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y 
conda install ipykernel -y 

# install transformers from source
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
cd examples/pytorch/text-classification/
pip install -r requirements.txt

cd ../../../../
rm -rf transformers

pip install datasets

conda install -c anaconda scikit-learn -y
conda install -c conda-forge ipywidgets -y

echo "=== instalation complete ==="

