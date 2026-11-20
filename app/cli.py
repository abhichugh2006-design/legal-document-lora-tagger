"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def cmd_prepare(args) -> None:
    from app import core
    from app.data import save_jsonl

    splits = core.prepare_dataset(args.source, per_template=args.per_template, seed=args.seed)
    for name, rows in splits.items():
        save_jsonl(rows, f"{args.out}/{name}.jsonl")
        print(f"{name:5s} {len(rows):5d} clauses -> {args.out}/{name}.jsonl")
    example = splits["train"][0]
    print("\nExample training sequence:\n" + example["prompt"] + example["target"])


def cmd_train(args) -> None:
    from app import core
    from app.data import TAGS
    from app.storage import Store

    splits = core.prepare_dataset(args.source, seed=args.seed)
    tags = TAGS if args.source is None else sorted({r["tag"] for r in splits["train"]})
    model, tokenizer = core.load_base(args.base_model)
    tag_ids = core.extend_tokenizer_and_model(model, tokenizer, tags)
    targets = tuple(t.strip() for t in args.targets.split(","))
    core.add_lora_adapter(model, tag_ids, r=args.rank, alpha=args.alpha, dropout=args.dropout, targets=targets)
    report = core.parameter_report(model)
    print(f"Trainable: {report['trainable_params']:,} of {report['base_params']:,} "
          f"({report['trainable_percent']:.3f}%, {report['reduction_factor']:.0f}x fewer)")

    history = core.train_tagger(model, tokenizer, tag_ids, splits, epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, seed=args.seed, device=args.device)
    config = {"base_model": args.base_model, "tags": tags, "r": args.rank, "alpha": args.alpha,
              "targets": list(targets), "epochs": args.epochs, "lr": args.lr, "seed": args.seed}
    size = core.save_adapter(model, args.out, config)
    print(f"Saved adapter ({size / 1e6:.2f} MB) to {args.out} in {history['train_seconds']:.0f}s "
          f"on {history['device']}")
    Store().put("last_train", {**config, "adapter": args.out, "adapter_bytes": size, **report,
                               "train_seconds": history["train_seconds"], "val_accuracy": history["val_accuracy"],
                               "final_loss": history["loss"][-1]})


def cmd_tag(args) -> None:
    import json
    from pathlib import Path

    from app import core

    model, tokenizer, tag_ids, config = core.load_tagger(args.adapter, args.device)
    merged = core.merge_adapter(model)
    print(f"Merged LoRA into {merged} layers; tagging {args.document}")
    blocks = core.tag_document(model, tokenizer, tag_ids, Path(args.document).read_text(), config["tags"],
                               review_threshold=args.review_threshold)
    report = core.render_tagged_document(blocks, f"Tagged: {Path(args.document).name}")
    out = Path(args.out or f"outputs/{Path(args.document).stem}.tagged.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    out.with_suffix(".json").write_text(json.dumps(blocks, indent=2))
    for n, b in enumerate([b for b in blocks if b["kind"] == "clause"], 1):
        flag = "  <- review" if b["needs_review"] else ""
        print(f"{n:3d}. {b['tag']:24s} {b['confidence']:5.0%}  {b['text'][:60]}...{flag}")
    print(f"\nWrote {out} and {out.with_suffix('.json')}")
    if args.save_merged:
        model.save_pretrained(args.save_merged)
        tokenizer.save_pretrained(args.save_merged)
        print(f"Saved merged GPT-2 to {args.save_merged}")


def cmd_evaluate(args) -> None:
    import gc
    import json
    from pathlib import Path

    from app import core
    from app.storage import Store

    splits = core.prepare_dataset(args.source)
    test = splits["test"]
    texts = [r["text"] for r in test]

    # Baseline: GPT-2 with the tag tokens added (name-initialised) but no fine-tuning at all.
    base, tokenizer = core.load_base(core.BASE_MODEL)
    saved_tags = core.torch.load(args.adapter, map_location="cpu", weights_only=True)["config"]["tags"]
    base_ids = core.extend_tokenizer_and_model(base, tokenizer, saved_tags)
    base.to(core.pick_device(args.device))
    baseline = core.evaluate_tagger(base, tokenizer, base_ids, test, saved_tags)
    del base
    gc.collect()

    model, tokenizer, tag_ids, config = core.load_tagger(args.adapter, args.device)
    tags = config["tags"]
    efficiency = core.parameter_report(model, tag_ids)
    efficiency["adapter_mb"] = Path(args.adapter).stat().st_size / 1e6
    efficiency["full_checkpoint_mb"] = efficiency["base_params"] * 4 / 1e6  # float32
    unmerged = core.evaluate_tagger(model, tokenizer, tag_ids, test, tags)
    efficiency["latency_unmerged_ms"] = core.measure_latency(model, tokenizer, tag_ids, texts)
    core.merge_adapter(model)
    merged = core.evaluate_tagger(model, tokenizer, tag_ids, test, tags)
    efficiency["latency_merged_ms"] = core.measure_latency(model, tokenizer, tag_ids, texts)
    agreement = sum(a == b for a, b in zip(unmerged["predictions"], merged["predictions"])) / len(test)

    result = {"config": config, "device": str(next(model.parameters()).device), "baseline": baseline,
              "lora": unmerged, "merged": merged, "merge_agreement": agreement, "efficiency": efficiency}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(core.render_evaluation(result))
    out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    Store().put("last_eval", {"accuracy": merged["accuracy"], "macro_f1": merged["macro_f1"],
                              "baseline_accuracy": baseline["accuracy"], **efficiency})
    print(f"Baseline (no fine-tuning): accuracy {baseline['accuracy']:.1%}, macro-F1 {baseline['macro_f1']:.3f}")
    print(f"LoRA:                      accuracy {unmerged['accuracy']:.1%}, macro-F1 {unmerged['macro_f1']:.3f}")
    print(f"LoRA merged:               accuracy {merged['accuracy']:.1%}, macro-F1 {merged['macro_f1']:.3f}  "
          f"(agreement with unmerged {agreement:.1%})")
    print(f"Trainable params {efficiency['trainable_params']:,} = {efficiency['trainable_percent']:.3f}% "
          f"({efficiency['reduction_factor']:.0f}x fewer); adapter {efficiency['adapter_mb']:.2f} MB vs "
          f"{efficiency['full_checkpoint_mb']:.0f} MB")
    print(f"Latency per clause: {efficiency['latency_unmerged_ms']:.2f} ms unmerged, "
          f"{efficiency['latency_merged_ms']:.2f} ms merged")
    print(f"Wrote {out} and {out.with_suffix('.json')}")


def add_lora_args(p) -> None:
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--targets", default="attn.c_attn", help="comma-separated layer names, e.g. attn.c_attn,attn.c_proj")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="legal-document-lora-tagger", description="Public sector clerks classify documents faster. This tool uses LoRA to fine-tune GPT-2 to tag legal clauses in public records, improving efficiency and reducing manual review time.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    p = sub.add_parser("prepare", help="milestone 1: build the clause dataset and write it as JSONL")
    p.add_argument("--source", help="folder with your own train/val/test.jsonl (default: built-in corpus)")
    p.add_argument("--per-template", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/clauses")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="milestones 2-4: add tag tokens, inject LoRA, fine-tune GPT-2")
    p.add_argument("--source", help="folder with your own train/val/test.jsonl (default: built-in corpus)")
    p.add_argument("--base-model", default="gpt2")
    add_lora_args(p)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", help="cpu, mps or cuda (default: best available)")
    p.add_argument("--out", default="artifacts/adapter.pt")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("tag", help="milestone 5: merge the adapter and tag every clause in a document")
    p.add_argument("document", help="plain-text file, clauses separated by blank lines or numbering")
    p.add_argument("--adapter", default="artifacts/adapter.pt")
    p.add_argument("--out", help="Markdown report path (default: outputs/<name>.tagged.md)")
    p.add_argument("--review-threshold", type=float, default=0.6)
    p.add_argument("--save-merged", help="also save the merged model to this folder")
    p.add_argument("--device")
    p.set_defaults(func=cmd_tag)

    p = sub.add_parser("evaluate", help="milestone 6: accuracy, F1 and efficiency on the held-out test set")
    p.add_argument("--adapter", default="artifacts/adapter.pt")
    p.add_argument("--source", help="folder with your own train/val/test.jsonl (default: built-in corpus)")
    p.add_argument("--out", default="outputs/evaluation.md")
    p.add_argument("--device")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Document LoRA Tagger is ready. Open README.md for your milestones.")
    else:
        args.func(args)


if __name__ == "__main__":
    main()
