# Legal Document LoRA Tagger

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ax9x3mxmwc/badge.svg)](https://ask.thili.ai/b/ax9x3mxmwc)

This project develops a 'Legal Document LoRA Tagger' to assist public sector clerks. By applying LoRA to fine-tune GPT-2, the tool efficiently tags legal clauses in public records, significantly reducing manual review time and improving classification accuracy. This innovation streamlines document processing in public services.

## What you are building

**Goal.** You will build a tool that fine-tunes GPT-2 using LoRA to classify and tag legal clauses in public records, making document review faster for public sector clerks.

**Done when.** You will have successfully fine-tuned GPT-2 with a LoRA adapter to tag legal clauses, and the final output will be a tagged legal document with evaluated accuracy and efficiency.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a dataset of legal clauses for tagging**  
      Hint: Format your legal clause data for fine-tuning, similar to how the 'Fine-Tune GPT-2...' drill prepares text for classification.
- [ ] **2. Extend GPT-2 tokenizer and model for custom tags**  
      Hint: Use the technique from 'Fine-Tune GPT-2...' to add special tokens for your tags and resize the model's embedding layer.
- [ ] **3. Implement LoRA adapter and inject into GPT-2**  
      Hint: Apply the 'Build LoRA from Scratch' technique to create and inject LoRA layers into GPT-2's attention blocks.
- [ ] **4. Fine-tune GPT-2 with LoRA on the clause dataset**  
      Hint: Combine the LoRA injection from 'Build LoRA from Scratch' with the fine-tuning process from 'Fine-Tune GPT-2...' on your clause dataset.
- [ ] **5. Merge LoRA adapter and perform inference on new clauses**  
      Hint: Use the merging technique from 'Build LoRA from Scratch' and perform inference on new legal texts to generate tags.
- [ ] **6. Evaluate tagging accuracy and model efficiency**  
      Hint: Calculate metrics like accuracy or F1-score on a test set, and consider the parameter count reduction from 'Build LoRA from Scratch' for efficiency.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different LoRA ranks and target layers to observe their impact on tagging accuracy and model efficiency.
- Compare the performance and resource usage of your LoRA-tuned model against a fully fine-tuned GPT-2 model (if computational resources allow for comparison).
- Analyze specific types of legal clauses where the model performs exceptionally well or struggles, suggesting areas for future dataset improvement.
