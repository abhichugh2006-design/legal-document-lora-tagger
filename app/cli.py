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

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Document LoRA Tagger is ready. Open README.md for your milestones.")
    else:
        args.func(args)


if __name__ == "__main__":
    main()
