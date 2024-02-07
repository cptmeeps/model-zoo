# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py

import os
import numpy as np
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import ModelArgs, build

# configs

log_interval = 10
eval_iters = 200 
init_from = 'scratch'
backend = 'nccl'
device = 'cuda' 
dtype = 'bfloat16' 
compile = True 
dataset = 'alpaca'
grad_accum = 4
batch_size = 12 
block_size = 1024
learning_rate = 6e-4 
max_iters = 600000 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 
decay_lr = True 
warmup_iters = 2000 
lr_decay_iters = 600000 
min_lr = 6e-5 

# helpers

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      with ctx:
        logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

def get_lr(it):
  if it < warmup_iters:
    return learning_rate * it / warmup_iters
  if it > lr_decay_iters:
    return min_lr
  decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
  return min_lr + coeff * (learning_rate - min_lr)

# init

init_process_group(backend=backend)
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'

torch.cuda.set_device(device)
master_process = ddp_rank == 0
seed_offset = ddp_rank 

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model_args = ModelArgs(max_batch_size=batch_size)
model = build(model_args)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
model = torch.compile(model) 
model = DDP(model, device_ids=[ddp_local_rank])

while True:
  lr = get_lr(iter_num) 
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  for micro_step in range(grad_accum):
    model.require_backward_grad_sync = (micro_step == grad_accum - 1)
    with ctx:
      logits = model.forward(X)
      logits = logits[:, -1]
      loss = loss_fn(logits, tgt)
      loss = loss / accum

    X, Y = get_batch('train')
    scaler.scale(loss).backward()
  
  if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
 
  scaler.step(optimizer)
  scaler.update()
  optimizer.zero_grad(set_to_none=True)

  if iter_num % log_interval == 0 and master_process:
    lossf = loss.item() * grad_accum_steps
    print(f"{iter_num} - {lossf:.4f}")

  iter_num += 1
  if iter_num > max_iters: break