"""
Run an evaluation on the whole validation data (1M tokens).
Should run in about 3 minutes.

"""

import time
from contextlib import nullcontext
import argparse

import torch
import torch._inductor.config as torch_ind_config

from models.lm import load_model
from data.dataloader import DataLoader

from utils.misc import format_time

# ------

vocab_size = 50257 # gpt2 enc
ctx_len = 1024

device = "cuda" # "cpu", "cuda:0", "cuda:1", ...
dtype = "bfloat16" # "float32" or "bfloat16"
use_torch_compile = True

# ------

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"
torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
dtype_ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, torch_dtype))

def main(load_dir, batch_size, verbose):
    model = load_model(load_dir, vocab_size, device)
    model.eval()

    if use_torch_compile:
        if hasattr(torch_ind_config, "coordinate_descent_tuning"):
            torch_ind_config.coordinate_descent_tuning = True

        model = torch.compile(model)

    val_loader = DataLoader("data/fineweb10B/fineweb_val_*.bin", batch_size, ctx_len, 1, 1)

    eval_loss = 0.0
    step = 0

    counter_tokens = 0
    num_tokens = len(val_loader.tokens)

    start_time = time.time()

    with torch.no_grad():
        while True:
            counter_tokens += batch_size * ctx_len
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            with dtype_ctx:
                loss = model(x, y)
            
            eval_loss += loss.item()
            
            # logging
            if verbose and step % 500 == 0:
                num_digits = len(str(num_tokens))
                print(f"[{int((100*counter_tokens/num_tokens)):02d}%] Processed {counter_tokens:0{num_digits}d}/{num_tokens:0{num_digits}d} tokens. uptime {format_time(time.time()-start_time)}.")

            step += 1

            if counter_tokens + batch_size*ctx_len >= num_tokens:
                break

            if step == 1:
                start_time = time.time() # actually start timing after the 1st iter (torch.compile)

    eval_loss /= step
    print(f"Evaluation loss: {eval_loss:.4f}")
    print(f"Took {time.time()-start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on the validation data.")
    parser.add_argument("--load_dir", type=str, required=True, help="Directory with a model.pth in it")
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--verbose", type=bool, required=False, default=False)

    args = parser.parse_args()

    print(f"Launching evaluation for {args.load_dir}")

    main(args.load_dir, args.batch_size, args.verbose)
