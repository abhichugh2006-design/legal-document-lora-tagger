"""The tagger, built milestone by milestone. Each section below is one milestone from README.md."""
from __future__ import annotations

from pathlib import Path

import torch

from app import lora
from app.data import TAGS, generate_clauses, load_jsonl

# =====================================================================================================
# Milestone 1: Prepare a dataset of legal clauses for tagging
# =====================================================================================================
# Like the GPT-2 classification drill, every example becomes one text sequence the model reads left to
# right:   "Clause: <clause text>\nTag:" + <tag token>
# The label is the single tag token after "Tag:". Every other position is masked (-100), so the loss
# measures only how well the model picks the tag.

PROMPT_PREFIX = "Clause: "
PROMPT_SUFFIX = "\nTag:"
MAX_LEN = 128
IGNORE = -100


def tag_token(tag: str) -> str:
    """Special token string for a tag, e.g. 'payment' -> '<|tag:payment|>'."""
    return f"<|tag:{tag}|>"


def format_prompt(text: str) -> str:
    return f"{PROMPT_PREFIX}{' '.join(text.split())}{PROMPT_SUFFIX}"


def format_example(text: str, tag: str) -> str:
    """The full training string: the prompt followed by the tag token."""
    return format_prompt(text) + tag_token(tag)


def prepare_dataset(source: str | None = None, per_template: int = 12, seed: int = 0) -> dict[str, list[dict]]:
    """Milestone 1. Return {'train', 'val', 'test'} lists of {'text', 'tag', 'prompt', 'target'}.

    `source=None` uses the built-in corpus (held-out templates for test). `source` can also be a folder
    with train.jsonl / val.jsonl / test.jsonl, each line {"text": ..., "tag": ...}.
    """
    if source is None:
        splits = generate_clauses(per_template=per_template, seed=seed)
    else:
        folder = Path(source)
        splits = {name: load_jsonl(folder / f"{name}.jsonl") for name in ("train", "val", "test")}
    for rows in splits.values():
        for row in rows:
            row["prompt"] = format_prompt(row["text"])
            row["target"] = tag_token(row["tag"])
    return splits


def encode_prompt(tokenizer, text: str, max_len: int = MAX_LEN) -> list[int]:
    """Token ids for the prompt. Long clauses are cut in the middle of the clause text, never in
    "Tag:", so the model always sees where the tag goes."""
    head = tokenizer.encode(PROMPT_PREFIX + " ".join(text.split()))
    tail = tokenizer.encode(PROMPT_SUFFIX)
    room = max_len - len(tail) - 1  # leave one position for the tag token
    return head[:room] + tail


def encode_example(tokenizer, text: str, tag: str, max_len: int = MAX_LEN) -> dict[str, list[int]]:
    """input_ids = prompt + tag token; labels are IGNORE except at the tag token."""
    prompt_ids = encode_prompt(tokenizer, text, max_len)
    tag_id = tokenizer.convert_tokens_to_ids(tag_token(tag))
    return {"input_ids": prompt_ids + [tag_id], "labels": [IGNORE] * len(prompt_ids) + [tag_id]}


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch of encoded examples into tensors."""
    width = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * (width - n))
        mask.append([1] * n + [0] * (width - n))
        if "labels" in b:
            labels.append(b["labels"] + [IGNORE] * (width - n))
    out = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
    if labels:
        out["labels"] = torch.tensor(labels)
    return out


def tag_counts(rows: list[dict]) -> dict[str, int]:
    counts = {tag: 0 for tag in TAGS}
    for row in rows:
        counts[row["tag"]] = counts.get(row["tag"], 0) + 1
    return counts


# =====================================================================================================
# Milestone 2: Extend GPT-2 tokenizer and model for custom tags
# =====================================================================================================
# Each tag becomes one new special token, so predicting a tag is a single next-token choice. After
# adding tokens, the embedding matrix is resized. GPT-2 ties its output head to the input embeddings,
# so each new row is both how the model reads the tag and the vector it scores against when it predicts
# one. The new rows start as the mean embedding of the tag's name (" force majeure", ...), which gives
# every tag a distinct, meaningful starting point instead of identical copies.

BASE_MODEL = "gpt2"


def load_base(model_name: str = BASE_MODEL):
    from transformers import AutoTokenizer, GPT2LMHeadModel

    return GPT2LMHeadModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)


def extend_tokenizer_and_model(model, tokenizer, tags: list[str] = TAGS) -> list[int]:
    """Milestone 2. Add one special token per tag, resize embeddings, initialise the new rows.

    Returns the token ids of the tags, in the same order as `tags`. Works in place.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token; masked out anyway
    name_ids = [tokenizer.encode(" " + tag.replace("_", " ")) for tag in tags]  # before adding tokens
    tokenizer.add_special_tokens({"additional_special_tokens": [tag_token(t) for t in tags]})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    tag_ids = tokenizer.convert_tokens_to_ids([tag_token(t) for t in tags])
    wte = model.get_input_embeddings().weight
    with torch.no_grad():
        for tag_id, ids in zip(tag_ids, name_ids):
            wte[tag_id] = wte[ids].mean(dim=0)
    model.config.pad_token_id = tokenizer.pad_token_id
    return tag_ids


# =====================================================================================================
# Milestone 3: Implement LoRA adapter and inject into GPT-2
# =====================================================================================================
# The LoRA layer itself is in app/lora.py. Here it is injected into GPT-2's attention blocks, and every
# pretrained weight is frozen. One more piece must learn: the new tag-token rows from milestone 2, or the
# model has no trained vector to score each tag against. A gradient mask lets only those few rows of
# the embedding matrix change; the original 50,257 rows stay frozen.

DEFAULT_TARGETS = ("attn.c_attn",)


def add_lora_adapter(model, tag_ids: list[int], r: int = 8, alpha: float = 16.0, dropout: float = 0.05,
                     targets=DEFAULT_TARGETS) -> list[str]:
    """Milestone 3. Freeze GPT-2, inject LoRA into `targets`, and unfreeze only the tag embedding rows."""
    names = lora.inject_lora(model, targets=targets, r=r, alpha=alpha, dropout=dropout)
    wte = model.get_input_embeddings().weight
    wte.requires_grad = True
    mask = torch.zeros(wte.shape[0], 1)
    mask[tag_ids] = 1.0
    wte.register_hook(lambda grad: grad * mask.to(grad.device, grad.dtype))
    model._tag_ids = list(tag_ids)
    return names


def parameter_report(model, tag_ids: list[int] | None = None) -> dict[str, float]:
    """Count what LoRA trains versus what full fine-tuning would train."""
    tag_ids = tag_ids if tag_ids is not None else getattr(model, "_tag_ids", [])
    n_lora = lora.count_lora_params(model)
    n_tag_rows = len(tag_ids) * model.get_input_embeddings().weight.shape[1]
    total = sum(p.numel() for p in model.parameters())
    base = total - n_lora
    trainable = n_lora + n_tag_rows
    return {"base_params": base, "lora_params": n_lora, "tag_embedding_params": n_tag_rows,
            "trainable_params": trainable, "trainable_percent": 100.0 * trainable / base,
            "reduction_factor": base / trainable if trainable else float("inf")}


# =====================================================================================================
# Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
# =====================================================================================================
# A plain PyTorch loop: AdamW over only the trainable tensors, linear warmup/decay, loss only on the tag
# token (milestone 1 labels). Validation accuracy is checked after each epoch. The saved adapter holds
# only the LoRA matrices and the tag rows: a few MB instead of a 500 MB GPT-2 checkpoint.


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def next_token_logits(model, input_ids, attention_mask, positions) -> torch.Tensor:
    """Full-vocabulary logits at one position per sequence, shape (batch, vocab).

    Same numbers as model(...).logits[range(batch), positions], but it skips projecting every position
    onto the 50k-token vocabulary, the largest tensor in a GPT-2 forward pass. The cross-entropy over
    these logits equals the usual shifted LM loss when only the tag token is labelled."""
    hidden = model.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return model.lm_head(hidden[torch.arange(hidden.shape[0], device=hidden.device), positions])


@torch.no_grad()
def tag_probabilities(model, tokenizer, tag_ids: list[int], texts: list[str], batch_size: int = 32,
                      max_len: int = MAX_LEN) -> torch.Tensor:
    """For each clause, a probability for every tag (the next-token distribution after "Tag:",
    restricted to the tag tokens). Shape (len(texts), len(tag_ids))."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(texts), batch_size):
        batch = [{"input_ids": encode_prompt(tokenizer, t, max_len)} for t in texts[start:start + batch_size]]
        enc = collate(batch, tokenizer.pad_token_id)
        last = enc["attention_mask"].sum(dim=1) - 1  # last real token, i.e. ":" in "Tag:"
        rows = next_token_logits(model, enc["input_ids"].to(device), enc["attention_mask"].to(device), last.to(device))
        out.append(rows[:, tag_ids].float().softmax(dim=-1).cpu())
    return torch.cat(out) if out else torch.empty(0, len(tag_ids))


def accuracy_on(model, tokenizer, tag_ids, rows: list[dict], tags: list[str] = TAGS) -> float:
    if not rows:
        return float("nan")
    probs = tag_probabilities(model, tokenizer, tag_ids, [r["text"] for r in rows])
    preds = [tags[i] for i in probs.argmax(dim=-1).tolist()]
    return sum(p == r["tag"] for p, r in zip(preds, rows)) / len(rows)


def train_tagger(model, tokenizer, tag_ids: list[int], splits: dict[str, list[dict]], epochs: int = 3,
                 batch_size: int = 16, lr: float = 3e-4, tag_lr: float = 2e-3, weight_decay: float = 0.01,
                 seed: int = 0, device: torch.device | str | None = None, max_steps: int | None = None,
                 log=print) -> dict:
    """Milestone 4. Train whatever is trainable in `model` (LoRA + tag rows, or everything for a full
    fine-tune) on splits['train']. Returns a history dict with losses, val accuracy and timing."""
    import random
    import time

    from transformers import get_linear_schedule_with_warmup

    device = pick_device(device) if not isinstance(device, torch.device) else device
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.to(device)
    wte = model.get_input_embeddings().weight
    others = [p for p in model.parameters() if p.requires_grad and p is not wte]
    groups = [{"params": others, "lr": lr, "weight_decay": weight_decay}]
    if wte.requires_grad:  # tag rows: no weight decay, or the frozen rows would shrink too
        groups.append({"params": [wte], "lr": tag_lr if hasattr(model, "_tag_ids") else lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups)

    encoded = [encode_example(tokenizer, r["text"], r["tag"]) for r in splits["train"]]
    steps_per_epoch = (len(encoded) + batch_size - 1) // batch_size
    total_steps = min(max_steps or 10**9, steps_per_epoch * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    history = {"loss": [], "val_accuracy": [], "device": str(device), "steps": 0}
    start = time.perf_counter()
    step = 0
    for epoch in range(epochs):
        model.train()
        rng.shuffle(encoded)
        for i in range(0, len(encoded), batch_size):
            batch = {k: v.to(device) for k, v in collate(encoded[i:i + batch_size], tokenizer.pad_token_id).items()}
            target_pos = (batch["labels"] != IGNORE).int().argmax(dim=1)  # where the tag token sits
            logits = next_token_logits(model, batch["input_ids"], batch["attention_mask"], target_pos - 1)
            loss = torch.nn.functional.cross_entropy(logits.float(), batch["labels"][torch.arange(len(target_pos)), target_pos])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            history["loss"].append(loss.item())
            step += 1
            if step % 10 == 0:
                log(f"epoch {epoch + 1} step {step}/{total_steps} loss {loss.item():.4f}")
            if step >= total_steps:
                break
        val_acc = accuracy_on(model, tokenizer, tag_ids, splits.get("val", []))
        history["val_accuracy"].append(val_acc)
        log(f"epoch {epoch + 1} done: val accuracy {val_acc:.3f}")
        if step >= total_steps:
            break
    history["steps"] = step
    history["train_seconds"] = time.perf_counter() - start
    model.eval()
    return history


def save_adapter(model, path: str | Path, config: dict) -> int:
    """Save only what LoRA trained. Returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tag_ids = model._tag_ids
    torch.save({"config": config, "lora": lora.lora_state_dict(model),
                "tag_rows": model.get_input_embeddings().weight[tag_ids].detach().cpu()}, path)
    return path.stat().st_size


def load_tagger(adapter_path: str | Path, device: str | None = None):
    """Rebuild GPT-2 + tag tokens + LoRA from a saved adapter. Returns (model, tokenizer, tag_ids, config)."""
    saved = torch.load(adapter_path, map_location="cpu", weights_only=True)
    config = saved["config"]
    model, tokenizer = load_base(config.get("base_model", BASE_MODEL))
    tag_ids = extend_tokenizer_and_model(model, tokenizer, config["tags"])
    add_lora_adapter(model, tag_ids, r=config["r"], alpha=config["alpha"], dropout=0.0,
                     targets=tuple(config["targets"]))
    missing = model.load_state_dict(saved["lora"], strict=False)
    if missing.unexpected_keys:
        raise ValueError(f"adapter does not match the model: {missing.unexpected_keys[:3]}")
    with torch.no_grad():
        model.get_input_embeddings().weight[tag_ids] = saved["tag_rows"]
    return model.to(pick_device(device)).eval(), tokenizer, tag_ids, config


# =====================================================================================================
# Milestone 5: Merge LoRA adapter and perform inference on new clauses
# =====================================================================================================
# Merging folds B @ A into each frozen weight (W <- W + alpha/r * B @ A), so the deployed model is a
# plain GPT-2 with no extra layers and no extra latency. Inference reads the next-token distribution
# after "Tag:" over the tag tokens only, which gives a confidence for each tag. A whole document is split
# into clauses, each clause is tagged, and low-confidence clauses are flagged for a clerk to review.

import re  # noqa: E402

REVIEW_THRESHOLD = 0.6


def merge_adapter(model) -> int:
    """Milestone 5. Fold every LoRA update into its base weight. Returns the number of merged layers."""
    return lora.merge_lora(model)


def predict(model, tokenizer, tag_ids: list[int], texts: list[str], tags: list[str] = TAGS,
            top_k: int = 3) -> list[dict]:
    """Tag each clause: {'tag', 'confidence', 'alternatives': [(tag, prob), ...]}."""
    probs = tag_probabilities(model, tokenizer, tag_ids, texts)
    results = []
    for row in probs:
        values, idx = row.topk(min(top_k, len(tags)))
        ranked = [(tags[i], round(v, 4)) for v, i in zip(values.tolist(), idx.tolist())]
        results.append({"tag": ranked[0][0], "confidence": ranked[0][1], "alternatives": ranked[1:]})
    return results


_NUMBERED = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]|\(?[a-z]\)|section\s+\d+|article\s+[\divx]+)", re.I)


def split_into_clauses(document: str, min_words: int = 6) -> list[dict]:
    """Split a document into blocks: paragraphs separated by blank lines, numbered items, and ALL-CAPS
    title lines. Titles, headings and party lines (short, or no closing punctuation) are kept as
    {'kind': 'heading'}; the rest are {'kind': 'clause'}."""
    blocks, current = [], []
    for raw in document.splitlines():
        line = raw.strip()
        starts_new = not line or _NUMBERED.match(line) or line.isupper() or (current and current[-1].isupper())
        if starts_new and current:
            blocks.append(" ".join(current))
            current = []
        if line:
            current.append(line)
    if current:
        blocks.append(" ".join(current))

    def kind(block: str) -> str:
        return "clause" if len(block.split()) >= min_words and block.rstrip()[-1] in ".;:" else "heading"

    return [{"kind": kind(b), "text": b} for b in blocks]


def tag_document(model, tokenizer, tag_ids: list[int], document: str, tags: list[str] = TAGS,
                 review_threshold: float = REVIEW_THRESHOLD) -> list[dict]:
    blocks = split_into_clauses(document)
    clauses = [b for b in blocks if b["kind"] == "clause"]
    for block, result in zip(clauses, predict(model, tokenizer, tag_ids, [c["text"] for c in clauses], tags)):
        block.update(result, needs_review=result["confidence"] < review_threshold)
    return blocks


def render_tagged_document(blocks: list[dict], title: str = "Tagged document") -> str:
    """A Markdown report: every clause with its tag, confidence, and a review flag, plus a summary."""
    clauses = [b for b in blocks if b["kind"] == "clause"]
    flagged = [b for b in clauses if b["needs_review"]]
    lines = [f"# {title}", "",
             f"{len(clauses)} clauses tagged, {len(flagged)} flagged for manual review "
             f"(confidence below {REVIEW_THRESHOLD:.0%}).", "", "| # | Tag | Confidence | Review |",
             "|---|-----|-----------|--------|"]
    for n, b in enumerate(clauses, 1):
        lines.append(f"| {n} | `{b['tag']}` | {b['confidence']:.0%} | {'check' if b['needs_review'] else ''} |")
    lines.append("")
    n = 0
    for b in blocks:
        if b["kind"] == "heading":
            lines += [f"## {b['text']}", ""]
            continue
        n += 1
        alts = ", ".join(f"{t} {p:.0%}" for t, p in b["alternatives"])
        flag = " **(needs review)**" if b["needs_review"] else ""
        lines += [f"**[{n}] `{b['tag']}`** {b['confidence']:.0%}{flag} <sub>next: {alts}</sub>", "",
                  f"> {b['text']}", ""]
    return "\n".join(lines)
