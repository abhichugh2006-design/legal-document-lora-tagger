"""Shared fixtures. Tests use the real GPT-2 tokenizer but a tiny randomly-initialised GPT-2, so the
whole suite runs in seconds on a laptop or in CI without downloading the 500 MB model weights."""
import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def tiny_gpt2(gpt2_tokenizer):
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=len(gpt2_tokenizer), n_positions=160, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(config).eval()
