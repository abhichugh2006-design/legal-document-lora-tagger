"""Tests for each milestone in README.md."""
import pytest

from app import core
from app.data import TAGS


# Milestone 1: Prepare a dataset of legal clauses for tagging
def test_milestone_1_dataset_is_split_formatted_and_held_out(gpt2_tokenizer):
    splits = core.prepare_dataset(per_template=6, seed=1)
    assert set(splits) == {"train", "val", "test"}
    for name, rows in splits.items():
        assert all(count > 0 for count in core.tag_counts(rows).values()), f"{name} is missing a tag"

    # Test clauses come only from templates never used for training.
    train_templates = {r["template"] for r in splits["train"] + splits["val"]}
    assert train_templates.isdisjoint({r["template"] for r in splits["test"]})

    row = splits["train"][0]
    assert row["prompt"].startswith("Clause: ") and row["prompt"].endswith("\nTag:")
    assert row["target"] == core.tag_token(row["tag"])
    assert core.format_example(row["text"], row["tag"]) == row["prompt"] + row["target"]

    # Same seed gives the same corpus.
    assert core.prepare_dataset(per_template=6, seed=1)["test"] == splits["test"]


def test_milestone_1_long_clauses_keep_the_tag_slot(gpt2_tokenizer):
    ids = core.encode_prompt(gpt2_tokenizer, "word " * 500, max_len=64)
    assert len(ids) == 63
    assert gpt2_tokenizer.decode(ids).endswith("\nTag:")


# Milestone 2: Extend GPT-2 tokenizer and model for custom tags
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Implement LoRA adapter and inject into GPT-2
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Merge LoRA adapter and perform inference on new clauses
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Evaluate tagging accuracy and model efficiency
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")
