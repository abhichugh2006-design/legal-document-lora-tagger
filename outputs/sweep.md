# Experiments: LoRA rank, target layers, full fine-tuning

Built-in corpus, held-out test templates (234 clauses), 3 epochs, seed 0, alpha = 2r, Apple M3 (8 GB, MPS).
One seed per row, so differences of a point or two are noise.

| Run | Trainable params | Saved weights | Test accuracy | Macro-F1 | Train time | GPU memory |
|---|---|---|---|---|---|---|
| LoRA r=1 on attn.c_attn | 44,544 | 0.2 MB | 99.1% | 0.992 | 44s | 3.5 GB |
| LoRA r=4 on attn.c_attn | 155,136 | 0.6 MB | 94.4% | 0.942 | 46s | 3.5 GB |
| LoRA r=8 on attn.c_attn | 302,592 | 1.2 MB | 91.9% | 0.917 | 64s | 4.6 GB |
| LoRA r=32 on attn.c_attn | 1,187,328 | 4.7 MB | 91.5% | 0.912 | 69s | 4.6 GB |
| LoRA r=8 on attn.c_attn,attn.c_proj | 450,048 | 1.8 MB | 94.0% | 0.938 | 59s | 3.5 GB |
| LoRA r=8 on attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj | 1,187,328 | 4.7 MB | 96.2% | 0.961 | 72s | 4.8 GB |
| Full fine-tuning (all weights) | 124,447,488 | 498 MB | out of memory | | | > 8 GB |

Full fine-tuning failed with `MPS ... Insufficient Memory` on this 8 GB laptop: gradients and AdamW state
for all 124M weights do not fit alongside the model. Every LoRA run fit in under 5 GB.
