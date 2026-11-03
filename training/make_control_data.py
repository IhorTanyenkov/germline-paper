"""Control data for the role-versus-training ablation.

Rewrites training/data/sft.jsonl so that every example renders the germline block the way the
untrained arm B2 sees it: at the head of the ordinary system message, with no `germline` role.
Everything else (user turns, assistant targets, slice tags, order) is byte-identical, so a model
trained on the output differs from the B3 adapter only in where the block sits.

    python3 -m training.make_control_data --inp training/data/sft.jsonl --out training/data/sft_sys.jsonl
"""
import argparse
import json


def fold(messages):
    if not messages or messages[0]["role"] != "germline":
        return messages, False
    germ = messages[0]["content"]
    rest = messages[1:]
    if rest and rest[0]["role"] == "system":
        folded = [{"role": "system", "content": germ + "\n\n" + rest[0]["content"]}] + rest[1:]
    else:
        folded = [{"role": "system", "content": germ}] + rest
    return folded, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="training/data/sft.jsonl")
    ap.add_argument("--out", default="training/data/sft_sys.jsonl")
    a = ap.parse_args()
    n = folded_n = 0
    with open(a.inp) as f, open(a.out, "w") as g:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["messages"], did = fold(row["messages"])
            folded_n += did
            n += 1
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{n} rows written to {a.out}; germline role folded into system in {folded_n}")


if __name__ == "__main__":
    main()
