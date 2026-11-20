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


    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Document LoRA Tagger is ready. Open README.md for your milestones.")
    else:
        args.func(args)


if __name__ == "__main__":
    main()
