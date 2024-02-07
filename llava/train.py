import torch
from PIL import Image
import random

class C3Dataset():
  def __init__(self, seq_len=32, bsz=4, shuffle=True):
    _, self.image_pre = load("ViT-L/14@336px")
    self.tokenizer = Tokenizer('tokenizer.model')
    self.seq_len = seq_len
    self.bsz = bsz

    with open('metadata.json', 'r') as file: 
      data = json.load(file)
    ds = [x for x in data if x.get('image') and x.get('blip_caption')]
    if shuffle: random.shuffle(ds)
    if bsz:
      ds = [ds[i:i + bsz] for i in range(0, len(ds) - len(ds) % bsz)] 
    self.ds = ds
    self.index = 0

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    batch = self.ds[idx]
    images = []
    all_tokens = []
    for data in batch:
      image = Image.open(f"dataset/cc3m/{data['image']}")
      image = self.image_pre(image).unsqueeze(0).to("cuda", dtype=torch.float32)
      prompt_tokens = self.tokenizer.encode(data['blip_caption'], bos=True, eos=False)
      tokens = [self.tokenizer.pad_id] * self.seq_len
      tokens[:len(prompt_tokens)] = prompt_tokens
      tokens = torch.tensor(tokens, dtype=torch.long, device="cuda")
      images.append(image)
      all_tokens.append(tokens)
    images_tensor = torch.cat(images, dim=0)
    tokens_tensor = torch.stack(all_tokens, dim=0)
    output = [tokens_tensor, images_tensor]
    return output

  def __iter__(self):
    self.index = 0
    return self

  def __next__(self):
    if self.index < len(self.ds):
      result = self.__getitem__(self.index)
      self.index += 1
      return result
    else:
      raise StopIteration