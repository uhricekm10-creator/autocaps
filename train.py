#!/usr/bin/env python3
"""Train a CLIP-prefix captioning model (ClipCap style).

A mapping network (transformer or MLP) turns a CLIP embedding into
`prefix_length` "prefix" token embeddings that are fed to GPT-2, which then
generates the caption.

Expected dataset (in --data_dir): one or more ``.pkl`` files, each a dict with
    "clip_embedding": FloatTensor [N, prefix_size]
    "captions":       list of {"caption": str, "clip_embedding": int}
where "clip_embedding" in each caption is the row index into the tensor.

Example:
    python3 train.py \
        --data_dir ./preprocessed_dataset/ \
        --output_dir ./checkpoints/ \
        --output_filename_prefix demo_model \
        --epochs 5 --save_every_epochs 1 --save_every_steps 10000 \
        --prefix_length 40 --prefix_size 512 \
        --mapping_type transformer --only_prefix \
        --num_layers 8 --num_heads 8 \
        --normalize_prefix --gpu_devices 0 \
        --cache_dir ./hf_cache          # any extra --key value goes to from_pretrained()
"""
import argparse
import glob
import os
import pickle
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Folder with preprocessed .pkl files")
    p.add_argument("--output_dir", default="./checkpoints/")
    p.add_argument("--output_filename_prefix", default="demo_model",
                   help="Checkpoint name prefix, e.g. 'demo_model_latest.ckpt'")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--save_every_epochs", type=int, default=1,
                   help="Save '<prefix>_<epoch>.ckpt' every N epochs (0 = off)")
    p.add_argument("--save_every_steps", type=int, default=10000,
                   help="Overwrite '<prefix>_latest.ckpt' every N steps (0 = off)")
    p.add_argument("--prefix_length", type=int, default=40)
    p.add_argument("--prefix_length_clip", type=int, default=None,
                   help="Transformer mapper only: number of tokens the CLIP vector is expanded to (default = prefix_length)")
    p.add_argument("--prefix_size", type=int, default=512, help="Dimension of the CLIP embeddings")
    p.add_argument("--mapping_type", choices=["transformer", "mlp"], default="transformer")
    p.add_argument("--only_prefix", action="store_true",
                   help="Train only the mapping network; keep the language model frozen")
    p.add_argument("--num_layers", type=int, default=8, help="Layers of the transformer mapper")
    p.add_argument("--num_heads", type=int, default=8, help="Attention heads of the transformer mapper")
    p.add_argument("--normalize_prefix", action="store_true", help="L2-normalize CLIP embeddings")
    p.add_argument("--use_deepspeed", action="store_true", help="[EXPERIMENTAL] train with DeepSpeed")
    p.add_argument("--gpu_devices", default="0",
                   help='"<n>", "<n>,<n>,..." or "-1" for all GPUs')
    p.add_argument("--language_model", default="gpt2")
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--max_caption_tokens", type=int, default=None,
                   help="Truncate captions to this many tokens (default: mean + 10*std, like ClipCap)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)  # set by deepspeed launcher

    args, unknown = p.parse_known_args(argv)
    args.huggingface_kwargs = parse_hf_kwargs(unknown)
    return args


def parse_hf_kwargs(tokens):
    """Turn leftover ``--key value`` / ``--flag`` / ``key=value`` into a kwargs dict."""
    kwargs, i = {}, 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            if "=" in key:
                key, val = key.split("=", 1)
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 1
                val = tokens[i]
            else:
                val = "true"
        elif "=" in tok:
            key, val = tok.split("=", 1)
        else:
            sys.exit(f"error: unexpected argument {tok!r}")
        kwargs[key.replace("-", "_")] = _coerce(val)
        i += 1
    return kwargs


def _coerce(val):
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    return val


def setup_devices(gpu_devices):
    # Must happen before CUDA is initialised.
    if gpu_devices.strip() != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices.strip()


ARGS = parse_args()
setup_devices(ARGS.gpu_devices)

import random  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          get_linear_schedule_with_warmup)


# --------------------------------------------------------------------------- data
class ClipCaptionDataset(Dataset):
    def __init__(self, data_dir, tokenizer, prefix_length, normalize_prefix, max_tokens=None):
        files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
        if not files:
            raise FileNotFoundError(f"No .pkl files found in {data_dir}")

        embeds, self.captions, self.embed_idx = [], [], []
        offset = 0
        for f in files:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            emb = torch.as_tensor(d["clip_embedding"]).float()
            if emb.dim() == 3:  # [N, 1, D]
                emb = emb.squeeze(1)
            embeds.append(emb)
            for c in d["captions"]:
                self.captions.append(c["caption"])
                self.embed_idx.append(offset + int(c["clip_embedding"]))
            offset += emb.shape[0]
            print(f"loaded {f}: {emb.shape[0]} embeddings, {len(d['captions'])} captions")

        self.prefixes = torch.cat(embeds, 0)
        if normalize_prefix:
            self.prefixes = self.prefixes / self.prefixes.norm(2, -1, keepdim=True).clamp_min(1e-8)
        self.prefix_length = prefix_length

        self.tokens = [torch.tensor(tokenizer.encode(c), dtype=torch.long) for c in self.captions]
        lens = torch.tensor([len(t) for t in self.tokens], dtype=torch.float)
        if max_tokens is None:
            max_tokens = int(min(lens.mean() + lens.std().nan_to_num(0) * 10, lens.max()))
        self.max_seq_len = max(1, max_tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        tok = self.tokens[i][: self.max_seq_len]
        pad = self.max_seq_len - tok.shape[0]
        mask = torch.cat([torch.ones(self.prefix_length + tok.shape[0]), torch.zeros(pad)])
        tok = torch.cat([tok, torch.zeros(pad, dtype=torch.long)])
        return tok, mask, self.prefixes[self.embed_idx[i]]


# --------------------------------------------------------------------------- model
class MLPMapper(nn.Module):
    def __init__(self, prefix_size, embed_dim, prefix_length):
        super().__init__()
        hidden = (embed_dim * prefix_length) // 2
        self.net = nn.Sequential(
            nn.Linear(prefix_size, hidden), nn.Tanh(), nn.Linear(hidden, embed_dim * prefix_length))
        self.prefix_length, self.embed_dim = prefix_length, embed_dim

    def forward(self, x):
        return self.net(x).view(-1, self.prefix_length, self.embed_dim)


class TransformerMapper(nn.Module):
    def __init__(self, prefix_size, embed_dim, prefix_length, clip_length, num_layers, num_heads):
        super().__init__()
        self.clip_length = clip_length
        self.linear = nn.Linear(prefix_size, clip_length * embed_dim)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 2,
                                           dropout=0.0, activation="relu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], -1, -1)
        out = self.transformer(torch.cat([x, prefix], dim=1))
        return out[:, self.clip_length:]


class ClipCaptionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.prefix_length = args.prefix_length
        self.lm = AutoModelForCausalLM.from_pretrained(args.language_model, **args.huggingface_kwargs)
        embed_dim = self.lm.get_input_embeddings().weight.shape[1]
        if args.mapping_type == "mlp":
            self.mapper = MLPMapper(args.prefix_size, embed_dim, args.prefix_length)
        else:
            if embed_dim % args.num_heads:
                raise ValueError(f"LM hidden size {embed_dim} not divisible by --num_heads {args.num_heads}")
            self.mapper = TransformerMapper(args.prefix_size, embed_dim, args.prefix_length,
                                            args.prefix_length_clip or args.prefix_length,
                                            args.num_layers, args.num_heads)
        self.only_prefix = args.only_prefix
        if self.only_prefix:
            for p in self.lm.parameters():
                p.requires_grad_(False)
            self.lm.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.only_prefix:
            self.lm.eval()
        return self

    def forward(self, tokens, prefix, mask):
        tok_emb = self.lm.get_input_embeddings()(tokens)
        prefix_emb = self.mapper(prefix).to(tok_emb.dtype)
        inputs = torch.cat([prefix_emb, tok_emb], dim=1)
        return self.lm(inputs_embeds=inputs, attention_mask=mask).logits


def caption_loss(logits, tokens, mask, prefix_length):
    # Position (prefix_length - 1 + t) predicts token t.
    logits = logits[:, prefix_length - 1:-1]
    tok_mask = mask[:, prefix_length:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1), reduction="none")
    tok_mask = tok_mask.reshape(-1)
    return (loss * tok_mask).sum() / tok_mask.sum().clamp_min(1)


# --------------------------------------------------------------------------- checkpoints
def trainable_state_dict(model):
    model = getattr(model, "module", model)
    sd = model.state_dict()
    if model.only_prefix:
        sd = {k: v for k, v in sd.items() if k.startswith("mapper.")}
    return sd


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, args):
    tmp = path + ".tmp"
    torch.save({
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,  # last fully completed epoch
        "step": step,
        "args": {k: v for k, v in vars(args).items() if k != "huggingface_kwargs"},
    }, tmp)
    os.replace(tmp, path)
    print(f"saved {path}")


# --------------------------------------------------------------------------- training
def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tok_kwargs = {k: v for k, v in args.huggingface_kwargs.items()
                  if k in ("cache_dir", "revision", "token", "local_files_only", "trust_remote_code")}
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, **tok_kwargs)
    dataset = ClipCaptionDataset(args.data_dir, tokenizer, args.prefix_length,
                                 args.normalize_prefix, args.max_caption_tokens)
    print(f"{len(dataset)} captions, max {dataset.max_seq_len} tokens each")

    model = ClipCaptionModel(args)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")

    steps_per_epoch = -(-len(dataset) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    start_epoch, global_step, ckpt = 0, 0, None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if unexpected:
            print(f"warning: unexpected keys in checkpoint: {unexpected[:5]}...")
        start_epoch, global_step = ckpt["epoch"] + 1, ckpt["step"]
        print(f"resumed from {args.resume} (epoch {ckpt['epoch']}, step {global_step})")

    if args.use_deepspeed:
        import deepspeed
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "AdamW", "params": {"lr": args.lr}},
            "scheduler": {"type": "WarmupDecayLR", "params": {
                "warmup_min_lr": 0, "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps, "total_num_steps": max(total_steps, 1)}},
            "fp16": {"enabled": torch.cuda.is_available()},
        }
        engine, optimizer, loader, scheduler = deepspeed.initialize(
            model=model, model_parameters=params, training_data=dataset, config=ds_config)
        device = engine.device
        is_main = engine.global_rank == 0
        fwd = engine
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        fwd = model
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            fwd = nn.DataParallel(model)
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, max(total_steps, 1))
        if ckpt is not None and ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        is_main = True
    print(f"device: {device}")

    latest = os.path.join(args.output_dir, f"{args.output_filename_prefix}_latest.ckpt")
    for epoch in range(start_epoch, args.epochs):
        fwd.train()
        bar = tqdm(loader, desc=f"epoch {epoch}", disable=not is_main)
        for tokens, mask, prefix in bar:
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device)
            logits = fwd(tokens, prefix, mask)
            loss = caption_loss(logits.float(), tokens, mask, args.prefix_length)

            if args.use_deepspeed:
                fwd.backward(loss)
                fwd.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            global_step += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

            if is_main and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                # mid-epoch save: record the last *completed* epoch so resume redoes this one
                save_checkpoint(latest, model, None if args.use_deepspeed else optimizer,
                                None if args.use_deepspeed else scheduler, epoch - 1, global_step, args)

        if is_main and args.save_every_epochs > 0 and (
                (epoch + 1) % args.save_every_epochs == 0 or epoch == args.epochs - 1):
            path = os.path.join(args.output_dir, f"{args.output_filename_prefix}_{epoch}.ckpt")
            save_checkpoint(path, model, None if args.use_deepspeed else optimizer,
                            None if args.use_deepspeed else scheduler, epoch, global_step, args)

    if is_main:
        save_checkpoint(latest, model, None if args.use_deepspeed else optimizer,
                        None if args.use_deepspeed else scheduler, args.epochs - 1, global_step, args)


if __name__ == "__main__":
    main(ARGS)
