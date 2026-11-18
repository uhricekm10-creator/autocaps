# autocaps
i need you to create a code for a claude that turns a voice into the autocaptions

## Training the caption model (`train.py`)

Trains a ClipCap-style model: a mapping network (transformer or MLP) turns a CLIP
embedding into prefix tokens for GPT-2, which writes the caption.

Install: `pip install torch transformers tqdm` (plus `deepspeed` for `--use_deepspeed`).

**Dataset** – `--data_dir` holds one or more `.pkl` files, each a dict:

```python
{"clip_embedding": FloatTensor[N, 512],          # one CLIP vector per image/clip
 "captions": [{"caption": "a dog runs", "clip_embedding": 0}, ...]}  # index into the tensor
```

**Run** (every line but the last ends in `\`; comments can't follow a `\`):

```bash
python3 train.py \
    --data_dir ./preprocessed_dataset/ \
    --output_dir ./checkpoints/ \
    --output_filename_prefix demo_model \
    --epochs 5 \
    --save_every_epochs 1 \
    --save_every_steps 10000 \
    --prefix_length 40 \
    --prefix_size 512 \
    --mapping_type transformer \
    --only_prefix \
    --num_layers 8 \
    --num_heads 8 \
    --normalize_prefix \
    --gpu_devices 0 \
    --cache_dir ./hf_cache
```

| Flag | Meaning |
|---|---|
| `--output_filename_prefix` | checkpoints are `<prefix>_<epoch>.ckpt` and `<prefix>_latest.ckpt` |
| `--save_every_epochs N` | write `<prefix>_<epoch>.ckpt` every N epochs |
| `--save_every_steps N` | overwrite `<prefix>_latest.ckpt` every N steps |
| `--prefix_size` | CLIP embedding size (512 for ViT-B/32) |
| `--mapping_type` | `transformer` or `mlp` |
| `--only_prefix` | train only the mapper, keep GPT-2 frozen |
| `--num_layers` / `--num_heads` | transformer mapper depth / attention heads |
| `--normalize_prefix` | L2-normalize CLIP embeddings |
| `--use_deepspeed` | [experimental] multi-GPU via DeepSpeed (launch with `deepspeed train.py ...`) |
| `--gpu_devices` | `"0"`, `"0,1"`, or `"-1"` for all |
| other `--key value` | passed to Hugging Face `from_pretrained` (e.g. `--cache_dir`, `--revision`, `--local_files_only true`) |

Also: `--batch_size` (40), `--lr` (2e-5), `--warmup_steps` (5000), `--language_model` (gpt2),
`--resume <ckpt>`. With `--only_prefix`, checkpoints store only the mapper weights.
