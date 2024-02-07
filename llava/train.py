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

class Dataset():
  def __init__(self, seq_len=32, bsz=4, shuffle=True):
    self.seq_len = seq_len
    self.bsz = bsz
    self.tkzr = Tokenizer("tokenizer.model")

    with open('alpaca.json', 'r') as file: 
      data = json.load(file)
    ds = data
    if shuffle: random.shuffle(ds)
    if bsz:
      bsz = bsz * 3
      ds = [ds[i:i + bsz] for i in range(0, len(ds) - len(ds) % bsz)] 

    self.ds = ds
    self.index = 0

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    tkzr = self.tkzr

    data = self.ds[idx]
    src_list = []
    tgt_list = []
    for d in data:
      src = "Below is an instruction that describes a task, paired with an inputthat provides "
      src += "further context. Write a response that appropriately completes the request."
      src += f"\n\n### Instruction:\n{d['instruction']}"
      src += f"\n\n### Input:\n{d['input']}"
      src += f"\n\n### Response:\n"
      if len(src) + len (d['output']) < 1024:
        src_list.append(src)
        tgt_list.append(d['output'])

    src = src_list
    tgt = tgt_list
    src_encoded = [tkzr.encode(t, bos=True, eos=False) for t in src]
    tgt_encoded = [tkzr.encode(t, bos=False, eos=True) for t in tgt]
    
    src_max_len = max(len(t) for t in src_encoded)
    src_normed, tgt_normed = [], []
    for s, t in zip(src_encoded, tgt_encoded):
      diff = src_max_len - len(s)
      if len(t[diff:]) > 6:
        s.extend(t[:diff])
        t = t[diff:]
        src_normed.append(s)
        tgt_normed.append(t)
    
    tgt_len = min(len(x) for x in tgt_normed if len(x) > 1)
    partition_point = random.randint(1, tgt_len - 2)
    src, tgt = [], []
    for s, t in zip(src_normed, tgt_normed):
      if len(t) > 0 and len(src) < self.bsz:
        s.extend(t[:partition_point])
        t = s[1:] + [t[partition_point]]
        src.append(s)
        tgt.append(t)
    src = torch.tensor(src)
    tgt = torch.tensor(tgt)
    return src, tgt


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


def finetune(train_len=6, bsz=4):
  tkzr = Tokenizer("tokenizer.model")
  model_args = ModelArgs(max_batch_size=bsz)
  model = build(model_args)
  # optimizer = optim.Adam(model.parameters(), lr=1e-6)
  optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
  optimizer.zero_grad()
  loss_fn = nn.CrossEntropyLoss()
  ds = Dataset(bsz=bsz)
  accum = 4 # 4
  for n in range(0, train_len):
    src, tgt = ds[n]
    logits = model.forward(src)
    logits = logits.view(-1, logits.size(-1))
    tgt = tgt.view(-1)
    loss = loss_fn(logits, tgt)
    loss_value = loss.item()
    loss = loss / accum
    loss.backward()
    if n % accum == 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      optimizer.zero_grad()
      print(f'{n},{loss_value}')

