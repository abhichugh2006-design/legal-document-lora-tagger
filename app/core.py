"""The tagger, built milestone by milestone. Each section below is one milestone from README.md."""
from __future__ import annotations

from pathlib import Path

import torch

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
