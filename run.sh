touch ~/.no_auto_tmux

sudo apt update && sudo apt upgrade -y

pip3 install numpy panda torch ftfy 
pip3 install regex sentencepiece 

sudo apt install git
git config --global user.name "cptmeep"
git config --global user.email "cptmeeps@gmail.com"

cd workspace

git clone https://github.com/cptmeeps/model-zoo

cd model-zoo

curl -o llava/ViT-L-14-336px.pt https://cptmeep-public.s3.us-west-2.amazonaws.com/model_zoo/clip/ViT-L-14-336px.pt
curl -o llava/bpe_simple_vocab_16e6.txt.gz https://cptmeep-public.s3.us-west-2.amazonaws.com/model_zoo/clip/bpe_simple_vocab_16e6.txt.gz

curl -o llava/tokenizer.model https://cptmeep-public.s3.us-west-2.amazonaws.com/model_zoo/llama-2-7b/tokenizer.model
curl -o llava/consolidated.00.pth https://cptmeep-public.s3.us-west-2.amazonaws.com/model_zoo/llama-2-7b/consolidated.00.pth

