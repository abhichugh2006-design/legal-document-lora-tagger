# Legal Document LoRA Tagger

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ax9x3mxmwc/badge.svg)](https://ask.thili.ai/b/ax9x3mxmwc)
[![tests](https://github.com/abhichugh2006-design/legal-document-lora-tagger/actions/workflows/ci.yml/badge.svg)](https://github.com/abhichugh2006-design/legal-document-lora-tagger/actions/workflows/ci.yml)

Public-sector clerks spend hours reading contracts and records to find the clauses that matter: who can
terminate, who pays, what must be disclosed under the public records law. This tool fine-tunes GPT-2 with a
**LoRA adapter written from scratch** to tag each clause of a document with one of ten clause types. Clauses
the model is unsure about are flagged so a clerk only has to review those.

```
$ python -m app.cli tag examples/sample_public_contract.txt
Merged LoRA into 12 layers; tagging examples/sample_public_contract.txt
  1. payment                   100%  1. Compensation. The District will pay the Consultant a not-...
  2. termination               100%  2. Early Termination. The District may end this Agreement at...
  3. confidentiality            92%  3. Protection of Patron Data. The Consultant shall not discl...
  4. public_records            100%  4. Open Records. The Consultant understands that the Distric...
  ...
 10. governing_law             100%  10. Applicable Law. This Agreement is governed by the laws o...
```

The full tagged report is in [outputs/sample_public_contract.tagged.md](outputs/sample_public_contract.tagged.md).

## Results

| | Value |
|---|---|
| Test accuracy (held-out phrasings) | **93.2%** (macro-F1 0.930) |
| GPT-2 with tag tokens but no fine-tuning | 52.6% (macro-F1 0.431) |
| Merged vs. unmerged predictions | identical on 100% of test clauses |
| Trainable parameters | **302,592 of 124.4M (0.243%), 411x fewer** than full fine-tuning |
| Saved adapter | **1.22 MB** vs. a 498 MB full checkpoint |
| Training time | 47 s on an Apple M3 laptop (MPS), 3 epochs |
| Inference | ~6 ms per clause |

The test set is written from **templates held out of training**, so it measures whether the model
recognises a clause type in new wording, not whether it memorised sentences. Full per-tag precision, recall,
F1, confusion matrix and error analysis: [outputs/evaluation.md](outputs/evaluation.md).

**Where it struggles.** All 16 test mistakes come from two phrasings:

- "Recovery … is restricted to direct damages, and the maximum exposure … is the contract value" is a
  limitation of liability, but it never says "liable" or "liability", so the model calls it
  indemnification.
- "Any non-public material … shall be treated as proprietary" is confidentiality, but "public" pulls it
  toward public records.

The model leans on keywords. Paraphrased training clauses without the obvious keyword would be the most
useful additions to the dataset.

## Run it

You need Python 3.12 and git. No API key is needed: everything runs locally. GPT-2 (500 MB) downloads from
Hugging Face on the first run.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.cli prepare                     # 1. build the clause dataset -> data/clauses/*.jsonl
python -m app.cli train                       # 2-4. add tag tokens, inject LoRA, fine-tune -> artifacts/adapter.pt
python -m app.cli tag examples/sample_public_contract.txt   # 5. merge + tag a document -> outputs/
python -m app.cli evaluate                    # 6. accuracy, F1, efficiency -> outputs/evaluation.md
pytest                                        # 14 tests, a few seconds (uses a tiny GPT-2)
```

Tag your own document: any plain-text file whose clauses are separated by blank lines or numbering
(`1.`, `2.1`, `(a)`, `Section 4`). The report goes to `outputs/<name>.tagged.md`, with a JSON copy.

Useful options: `train --rank 16 --targets attn.c_attn,attn.c_proj --epochs 5`,
`tag --review-threshold 0.8`, `tag --save-merged artifacts/merged` (saves a standard Hugging Face model
folder), and `--device cpu` on any command.

**Your own data.** Put `train.jsonl`, `val.jsonl` and `test.jsonl` in a folder, one
`{"text": "...", "tag": "..."}` per line, then pass `--source that/folder` to `train` and `evaluate`.

The `.env` file from the starter is only for the optional chat helper in `app/llm.py`. The tagger does not
use it.

## How it works

| Step | Code | What happens |
|---|---|---|
| 1. Dataset | [app/data.py](app/data.py), `prepare_dataset` | 100 hand-written clause templates (10 per tag) across termination, confidentiality, indemnification, governing law, payment, limitation of liability, public records, insurance, non-discrimination and force majeure. Slots (agency, vendor, days, amounts, state, records act) are filled at random with a fixed seed: 718 train / 126 val / 234 test. The last 2 templates of every tag are used **only** for test. Each example becomes `Clause: <text>\nTag:<|tag:payment|>`. |
| 2. Tag tokens | `extend_tokenizer_and_model` | Adds 10 special tokens (`<|tag:payment|>`, ...) and resizes GPT-2's embeddings. GPT-2 ties the input embeddings to the output head, so each new row is also the vector the model scores against when it predicts that tag. New rows start as the mean embedding of the tag's name, e.g. " force majeure". |
| 3. LoRA | [app/lora.py](app/lora.py), `add_lora_adapter` | `LoRALayer` wraps GPT-2's `Conv1D` (weight stored as in×out): `y = xW + b + (α/r)·x·Aᵀ·Bᵀ`, with A Kaiming-initialised and B zero, so the model is unchanged at the start. All of GPT-2 is frozen. Only A, B and the 10 new tag rows train; a gradient mask keeps the 50,257 original embedding rows fixed. |
| 4. Fine-tune | `train_tagger` | AdamW with linear warmup/decay. Loss is cross-entropy on the tag token only, and logits are computed at that single position instead of across all 50k-vocabulary × sequence positions, which uses about 80x less activation memory. A test checks it gives the same loss as the standard Hugging Face loss. |
| 5. Merge + infer | `merge_adapter`, `tag_document` | `W ← W + (α/r)·(BA)ᵀ` folds each adapter into its frozen weight and removes the LoRA layers, leaving a plain GPT-2 with no extra inference cost. The tag is the most likely tag token after `Tag:`; its probability is the confidence, and anything under 60% is flagged for review. |
| 6. Evaluate | `evaluate_tagger`, `render_evaluation` | Accuracy, macro-F1, per-tag P/R/F1 and a confusion matrix against an untrained baseline, plus parameter counts, file sizes and latency before and after merging. |

## Experiments (stretch)

`python -m app.cli sweep --full` trains each configuration from scratch and scores it on the same held-out
test set ([outputs/sweep.md](outputs/sweep.md)):

| Run | Trainable params | Saved weights | Test accuracy | Macro-F1 | Train time (M3) |
|---|---|---|---|---|---|
| LoRA r=1 on attn.c_attn | 44,544 | 0.18 MB | 99.1% | 0.992 | 44s |
| LoRA r=4 on attn.c_attn | 155,136 | 0.62 MB | 94.4% | 0.942 | 46s |
| LoRA r=8 on attn.c_attn (default) | 302,592 | 1.2 MB | 91.9% | 0.917 | 64s |
| LoRA r=32 on attn.c_attn | 1,187,328 | 4.7 MB | 91.5% | 0.912 | 69s |
| LoRA r=8 on attn.c_attn + attn.c_proj | 450,048 | 1.8 MB | 94.0% | 0.938 | 59s |
| LoRA r=8 on attention + MLP (all 4 layers) | 1,187,328 | 4.7 MB | 96.2% | 0.961 | 72s |

One seed per row on a 234-clause test set. The same r=8 setup scored 93.2% in the main run and 91.9%
here (MPS training is not bit-for-bit deterministic), so treat gaps under about 2 points as noise.

**What it shows.** A higher rank does *not* help: r=1, with 44k parameters (0.04% of GPT-2), did as well as
or better than r=32. Telling apart ten clause types is a low-rank change, and extra capacity mostly fits the
training templates. Spreading the same rank across more layers (attention + MLP) helped more than raising
the rank on one layer. **Full fine-tuning did not fit.** `sweep --full` (all 124M weights trainable) ran out of GPU memory on this 8 GB M3 laptop: gradients plus AdamW state for every weight take several GB more than the model itself, while every LoRA run above trained in under 5 GB. On this hardware LoRA is the only way to fine-tune GPT-2, and its adapter is 100-2,800x smaller to store. On a machine with more memory, the same command adds the full fine-tuning row to `outputs/sweep.md`.

## Milestones

- [x] **1. Prepare a dataset of legal clauses for tagging**: `prepare_dataset`, `python -m app.cli prepare`
- [x] **2. Extend GPT-2 tokenizer and model for custom tags**: `extend_tokenizer_and_model`
- [x] **3. Implement LoRA adapter and inject into GPT-2**: `app/lora.py`, `add_lora_adapter`
- [x] **4. Fine-tune GPT-2 with LoRA on the clause dataset**: `train_tagger`, `python -m app.cli train`
- [x] **5. Merge LoRA adapter and perform inference on new clauses**: `merge_adapter`, `python -m app.cli tag`
- [x] **6. Evaluate tagging accuracy and model efficiency**: `evaluate_tagger`, `python -m app.cli evaluate`

Stretch: rank and target-layer sweep, full fine-tuning comparison, and per-clause-type error analysis (above).

## Limitations

- The built-in corpus is generated from templates, so it is cleaner and more uniform than real public
  records. Treat 93% as accuracy on unseen *phrasings* of familiar clause types, not on real documents.
  Retrain with `--source` on labelled real clauses before relying on it.
- Each clause gets exactly one tag; real clauses sometimes mix two (e.g. termination for force majeure).
- GPT-2 sees at most 128 tokens per clause; longer clauses are cut in the middle, keeping the `Tag:` prompt.
- It is a review aid, not legal advice: a clerk should check flagged clauses and anything high-stakes.

## Project layout

```
app/data.py      clause templates, corpus generation, JSONL I/O
app/lora.py      LoRA from scratch: LoRALayer, inject_lora, merge_lora
app/core.py      the six milestones, in order
app/cli.py       prepare / train / tag / evaluate / sweep
tests/           one test (or more) per milestone, using a tiny GPT-2
examples/        a sample public-sector contract to tag
outputs/         the tagged sample, evaluation report and sweep results
```
