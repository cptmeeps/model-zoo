import os, math, time, random, json
import requests
from io import BytesIO
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple, List
from sentencepiece import SentencePieceProcessor
import torch
from torch import nn, tensor
import torch.autograd.profiler as profiler
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from clip import load

    # [print(tkzr.decode(x)) for x in src_txt.tolist()]
    # print('tgt:', tkzr.decode(tgt.tolist())) 
# print('src_txt, src_img, tgt', src_txt.shape, src_img.shape, tgt.shape)


# llama

@dataclass
class ModelArgs:
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: Optional[int] = None
  vocab_size: int = 32000
  multiple_of: int = 256  
  ffn_dim_multiplier: Optional[float] = None
  norm_eps: float = 1e-5
  max_batch_size: int = 32
  max_seq_len: int = 1024
  img_len: int = 577
  img_dim: int = 1024
  max_gen_len: int = 32

class Tokenizer:
  def __init__(self, model_path: str):
    assert os.path.isfile(model_path), model_path
    self.sp_model = SentencePieceProcessor(model_file=model_path)

    self.n_words: int = self.sp_model.vocab_size()
    self.bos_id: int = self.sp_model.bos_id()
    self.eos_id: int = self.sp_model.eos_id()
    self.pad_id: int = self.sp_model.pad_id()
    assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

  def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    assert type(s) is str
    t = self.sp_model.encode(s)
    if bos:
      t = [self.bos_id] + t
    if eos:
      t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
    return self.sp_model.decode(t)

class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

class MLP(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.w1 = nn.Linear(args.img_dim, args.dim, bias=True).to(dtype=torch.float16)
    self.gelu = nn.GELU()
    self.w2 = nn.Linear(args.dim, args.dim, bias=True).to(dtype=torch.float16)

  def forward(self, x):
    h = self.w1(x)
    h = self.gelu(h)
    h = self.w2(h)
    return h

class Attention(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.n_heads = args.n_heads
    self.head_dim = args.dim // args.n_heads
    self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  

  def reshape_for_broadcast(self, freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

  def apply_rotary_emb(self, xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

  def forward(self, x, freqs_cis, mask=None):
    # print('Attention.forward', x.shape, x.device, x.dtype)
    bsz, seqlen, _ = x.shape
    # apply linear layer with checkpointing
    xq = checkpoint(lambda x: self.wq(x), x)
    xk = checkpoint(lambda x: self.wk(x), x)
    xv = checkpoint(lambda x: self.wv(x), x)
    # reshape to heads and head dims
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
    # apply rotary encoding
    xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    # transpose heads to 2D for easier computation
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    # dot product of q v, scaled by the sqrt of head dims
    dot_product = torch.matmul(xq, xk.transpose(2, 3))
    scores = dot_product / math.sqrt(self.head_dim)
    # apply masks
    if mask is not None:
      scores = scores + mask
    # softmax to get attn scores
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    # get weighted sum from scores and value
    output = torch.matmul(scores, xv)
    # reshape back to original shape, apply linear layer
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x):
    x = checkpoint(lambda x: F.silu(self.w1(x)) * self.w3(x), x)
    return self.w2(x)

class TransformerBlock(nn.Module):
  def __init__(self, layer_id: int, args):
    super().__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads
    self.attention = Attention(args)
    self.feed_forward = FeedForward(
      dim=args.dim,
      hidden_dim=4 * args.dim,
      multiple_of=args.multiple_of,
      ffn_dim_multiplier=args.ffn_dim_multiplier,
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

  def forward(self, x, freqs_cis, mask=None, ):
    # print('TransformerBlock.forward', x.shape, x.device, x.dtype)
    h = x + self.attention.forward(
      self.attention_norm(x), freqs_cis, mask
    )
    out = h + self.feed_forward.forward(self.ffn_norm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.clip, _ = load("ViT-L/14@336px")
    self.mlp = MLP(params)
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers
    self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
    self.layers = torch.nn.ModuleList()
    self.freqs_cis = self.precompute_freqs_cis(
      self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
    )
    
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(layer_id, params))
    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

  def precompute_freqs_cis(self, dim, end, theta=10000.0):
    indices = torch.arange(0, dim, 2) 
    sliced_indices = indices[: (dim // 2)].float()
    scaled_indices = sliced_indices / dim
    theta_power = theta ** scaled_indices
    freqs = 1.0 / theta_power
    t = torch.arange(end, device="cuda")
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

  # @torch.inference_mode()
  def forward(self, toks, imgs=None): 
    image = None
    _bsz, seqlen = toks.shape
    h = self.tok_embeddings(toks)
    
    if imgs != None:
      image_encoded = self.clip.encode_image(imgs).to(h.device)#, dtype=torch.float16)
      image_encoded.detach()
      image_projected = self.mlp(image_encoded)
      image_projected = image_projected.expand(_bsz, -1, -1)
      # print('image_projected.shape', image_projected.shape)
      # print('image_projected: ', image_projected[0, 1, :])
      seqlen += image_projected.size(1)
      # print('h after: ', h[0, 0, :])
      h_before = h[:, :1, :]
      h_after = h[:, 1:, :]
      h = torch.cat([h_before, image_projected, h_after], dim=1)
      # print('h after: ', h[0, 0, :])
    
    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[:seqlen]
    mask = None
    
    if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"), device=toks.device)
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([
        torch.zeros((seqlen, 0), device=toks.device), mask
      ]).type_as(h)
    
    for layer in self.layers:
      h = layer(h, freqs_cis, mask)
    h = self.norm(h)
    output = self.output(h).float()
    return output

# model utils

def build(model_args):
  tkzr = Tokenizer("tokenizer.model")
  torch.cuda.set_device(0)
  torch.set_default_tensor_type(torch.cuda.HalfTensor)  
  ckpt = torch.load("consolidated.00.pth", map_location="cuda")
  model = Transformer(model_args)
  model.load_state_dict(ckpt, strict=False)
  print('model loaded')
  return model

def generate(self, model, tkzr, model_args, tkns, image=None, max_gen=32):
  start_time = time.time()
  bsz = len(tkns)
  if max_gen:
    max_gen_len = max_gen
  else:  
    max_gen_len = model_args.max_seq_len - 1
  min_tkn_len = min(len(t) for t in tkns)
  max_tkn_len = max(len(t) for t in tkns)
  ttl_len = min(model_args.max_seq_len, max_gen_len + max_tkn_len)

  pad_id = tkzr.pad_id
  gen_tkns = torch.full((bsz, ttl_len), pad_id, dtype=torch.long)
  for k, t in enumerate(tkns):
    gen_tkns[k, : len(t)] = torch.tensor(t, dtype=torch.long)
  eos_reached = torch.tensor([False] * bsz)
  input_text_mask = gen_tkns != pad_id
  
  for cur_pos in range(min_tkn_len, ttl_len):
    logits = model.forward(gen_tkns[:, :cur_pos])#, image)
    nxt_tkn = torch.argmax(logits[:, -1], dim=-1)
    nxt_tkn = nxt_tkn.reshape(-1)
    nxt_tkn = torch.where(input_text_mask[:, cur_pos], gen_tkns[:, cur_pos], nxt_tkn)
    gen_tkns[:, cur_pos] = nxt_tkn
    eos_reached |= (~input_text_mask[:, cur_pos]) & (nxt_tkn == tkzr.eos_id)
    if all(eos_reached): break

  out_tkns = []
  for i, t in enumerate(gen_tkns.tolist()):
    tkns_len = len(tkns[i])
    t = t[tkns_len : tkns_len + max_gen_len]
    if tkzr.eos_id in t:
      eos_idx = t.index(tkzr.eos_id)
      t = t[:eos_idx]
    out_tkns.append(t)  
  print(f"generated in {time.time() - start_time:.2f} seconds")
  return out_tkns

def test(self):
  tkzr = Tokenizer("tokenizer.model")
  prompts = {
    'txt' : [
      "Simply put, the theory of relativity states that", # the laws of physics are the same for all non-accelerating observers, regardless of their state of motion or their energy content.
      "Long ago there lived a magical cat named Puss" # in Boots. Puss in Boots was a very clever cat. He was so clever that he could talk. He was so clever that he could talk
    ],
    'img' : ["Simply put, the theory of relativity states that"],
    'train' : [
      "image label:", "image label:",
      "photo description:", "photo description:",
      "image title:", "image title:",
      "picture summary:", "picture summary:",
    ]
  }
  model = build() # max_batch_size=4, max_seq_len=32
  tkns = [tkzr.encode(x, bos=True, eos=False) for x in prompts['txt']]
  text_out_tkns = generate(tkns, image=None)
  [print(tkzr.decode(t)) for t in text_out_tkns]

# train

class Dataset():
  def __init__(self, seq_len=32, bsz=4, shuffle=True):
    self.clip, self.image_pre = load("ViT-L/14@336px")
    self.tkzr = Tokenizer('tokenizer.model')
    self.seq_len = seq_len

    self.prompts = ["image label: ", "photo description: ","image title: ", "picture summary: "]

    with open('data/metadata.json', 'r') as file: 
      data = json.load(file)
    ds = [x for x in data if x.get('image') and x.get('blip_caption')]
    if shuffle: random.shuffle(ds)
    self.bsz_mult = 3
    self.bsz = bsz * self.bsz_mult
    ds = [ds[i:i + self.bsz] for i in range(0, len(ds) - len(ds) % self.bsz)] 
    self.ds = ds
    self.index = 0

  def __len__(self):
    return len(self.ds)

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

  def __getitem__(self, idx):
    batch = self.ds[idx]
    text_toks = [self.tkzr.encode(d['blip_caption'], bos=False, eos=False) for d in batch]
    min_len = min(len(x) for x in text_toks)

    _images = []
    _text = []
    _target = []
    for data in batch:
      if len(_images) == self.bsz / self.bsz_mult:
        break
      img = Image.open(f"data/images/{data['image']}")
      img = self.image_pre(img).unsqueeze(0).to("cuda")
      _images.append(img)

      txt = data['blip_caption']
      prefix = random.choice(self.prompts)
      txt = prefix + txt
      txt = self.tkzr.encode(data['blip_caption'], bos=True, eos=False)
      txt = txt[: min_len - 2]
      _target.append(txt[-1])
      txt = txt[:-1]
      txt = torch.tensor(txt, dtype=torch.long, device="cuda")
      _text.append(txt)
      
    tgt = torch.tensor(_target, dtype=torch.long, device="cuda")
    src_img = torch.cat(_images, dim=0)
    src_txt = torch.stack(_text, dim=0)
    return src_txt, src_img, tgt

def _train(bsz=4):
  _, img_pre = load("ViT-L/14@336px")
  tokenizer = Tokenizer("tokenizer.model")
  model_args = ModelArgs(max_batch_size=4)
  model = build(model_args)

  learning_rate = 0.01
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  criterion = nn.CrossEntropyLoss()

  ds = C3MDS(bsz=bsz)
  for n in range(0,50):
    imgs, txts = ds[n]
    imgs = [img_pre(i).unsqueeze(0) for i in imgs]
    imgs = torch.cat(imgs, dim=0).to("cuda", dtype=torch.float32)
    caption = [tokenizer.encode(t, bos=False, eos=False) for t in txts]

    min_len = min([len(t) for t in caption])
    if min_len < 3: continue
    cutoff = random.randint(1, min_len)
    src = [x[:cutoff][:-1] for x in caption]
    src = torch.tensor(src, dtype=torch.long, device="cuda")
    prefix = prompts['train']
    random.shuffle(prefix)
    prefix = prefix[:bsz]
    pre_prompt = [tokenizer.encode(x, bos=False, eos=False) for x in prefix]
    pre_prompt = torch.tensor(pre_prompt, dtype=torch.long, device="cuda")
    src = torch.cat((pre_prompt, src), dim=1)

    tgt = [x[:cutoff][-1] for x in caption]
    tgt = torch.tensor(tgt, dtype=torch.long, device="cuda")

    logits = model.forward(src, imgs)
    loss = criterion(logits[:, -1, :], tgt)
    print(f'{n}\t{loss.item()}')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def train(train_len=20, bsz=3, accum=4):
  _, img_pre = load("ViT-L/14@336px")
  tkzr = Tokenizer("tokenizer.model")
  model_args = ModelArgs(max_batch_size=bsz)
  model = build(model_args)

  for name, m in model.mlp.named_modules():
    if isinstance(m, nn.Linear):
      init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        init.constant_(m.bias, 0)
      print(f"kaiming normal init: {name}")

  for name, param in model.named_parameters():
    if 'clip' in name:
      param.requires_grad = False
    if 'layers' in name:
      param.requires_grad = False
    if name in ['norm.weight', 'output.weight', 'tok_embeddings.weight']:
      param.requires_grad = False

  for name, param in model.named_parameters():
    if param.requires_grad:
      if "weight" in name:
        pass
        print(f"{name} - row\n{param.data[0]}")
        print(f"{name} - col\n{param.data[:, 0]}")

  optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
  # optimizer.zero_grad()
  loss_fn = nn.CrossEntropyLoss()
  ds = Dataset(bsz=bsz)
  gradient_accumulators = {}

  for n in range(1, train_len):
    src_txt, src_img, tgt = ds[n]
    # print('\n src_txt, src_img, tgt', src_txt.shape, src_img.shape, tgt.shape)
    # with profiler.profile(use_cuda=torch.cuda.is_available(), record_shapes=True) as prof:
    logits = model.forward(src_txt, src_img)
      # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    split_idx = src_txt.shape[1] - 1
    if split_idx == 0: continue
    logits = logits[:, -split_idx:, :]
    logits = logits.reshape(-1, logits.size(-1))
    
    tgt_prefix = src_txt[:, 2:]
    tgt = tgt.unsqueeze(1)
    tgt = torch.cat((tgt_prefix, tgt), dim=1)
    tgt = tgt.reshape(-1)
    
    loss = loss_fn(logits, tgt)
    loss_value = loss.item()
    loss = loss / accum

    loss.backward()

    # accumulate gradients for each layer
    for name, param in model.named_parameters():
      if param.requires_grad:
        if "weight" in name:
          if name not in gradient_accumulators:
            gradient_accumulators[name] = param.grad.clone()
          else:
            gradient_accumulators[name] += param.grad
    
    if n % accum == 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      optimizer.zero_grad()
      print(f'\n{n}\t{loss_value}')
      
      # display accumulated gradients
      for name, accumulated_grad in gradient_accumulators.items():
        print(f"{name}: {accumulated_grad.norm().item()}")  
      gradient_accumulators = {}  

















train(train_len=10, bsz=8, accum=2)



