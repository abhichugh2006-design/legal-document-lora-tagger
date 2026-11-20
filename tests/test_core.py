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
def test_milestone_2_tags_are_single_tokens_with_distinct_embeddings(tiny_gpt2, gpt2_tokenizer):
    import copy

    tok = copy.deepcopy(gpt2_tokenizer)
    old_vocab = len(tok)
    tag_ids = core.extend_tokenizer_and_model(tiny_gpt2, tok)

    assert len(tok) == old_vocab + len(TAGS)
    assert tiny_gpt2.get_input_embeddings().weight.shape[0] == len(tok)
    assert tiny_gpt2.get_output_embeddings().weight.shape[0] == len(tok)  # tied head resized too
    assert tag_ids == list(range(old_vocab, old_vocab + len(TAGS)))
    for tag, tag_id in zip(TAGS, tag_ids):
        assert tok.encode(core.format_example("x", tag))[-1] == tag_id  # tag is one token
    rows = tiny_gpt2.get_input_embeddings().weight[tag_ids]
    assert len({tuple(r.tolist()) for r in rows}) == len(TAGS)  # all different starting points
    assert tok.pad_token is not None

    encoded = core.encode_example(tok, "The City shall pay.", "payment")
    assert encoded["labels"][-1] == tag_ids[TAGS.index("payment")]
    assert all(label == core.IGNORE for label in encoded["labels"][:-1])


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
