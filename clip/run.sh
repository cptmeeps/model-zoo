touch ~/.no_auto_tmux;

sudo apt update && sudo apt upgrade -y

pip3 install numpy panda regex
pip3 install sentencepiece ftfy
pip3 install torch

sudo apt install git && \
git config --global user.name "cptmeep" && \
git config --global user.email "cptmeeps@gmail.com"

curl -o ViT-L-14-336px.pt https://cptmeep-public.s3.us-west-2.amazonaws.com/model_zoo/clip/ViT-L-14-336px.pt
curl -o clip.py https://cptmeep-public.s3.us-west-2.amazonaws.com/model_zoo/clip/clip.py
curl -o bpe_simple_vocab_16e6.txt.gz https://cptmeep-public.s3.us-west-2.amazonaws.com/model_zoo/clip/bpe_simple_vocab_16e6.txt.gz
