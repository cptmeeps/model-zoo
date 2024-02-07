touch ~/.no_auto_tmux;

sudo apt update && sudo apt upgrade -y

pip3 install numpy panda regex
pip3 install sentencepiece ftfy
pip3 install torch

sudo apt install git && \
git config --global user.name "cptmeep" && \
git config --global user.email "cptmeeps@gmail.com"

curl -o tokenizer.model https://cptmeep-public.s3.us-west-2.amazonaws.com/model_zoo/llama-2-7b/tokenizer.model
curl -o alpaca.json https://cptmeep-public.s3.us-west-2.amazonaws.com/datasets/alpaca_data_cleaned.json
curl -o consolidated.00.pth https://cptmeep-public.s3.us-west-2.amazonaws.com/model_zoo/llama-2-7b/consolidated.00.pth