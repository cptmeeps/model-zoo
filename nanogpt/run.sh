touch ~/.no_auto_tmux;

sudo apt update && sudo apt upgrade -y

pip3 install numpy panda regex
pip3 install sentencepiece ftfy
pip3 install torch

sudo apt install git && \
git config --global user.name "cptmeep" && \
git config --global user.email "cptmeeps@gmail.com"
