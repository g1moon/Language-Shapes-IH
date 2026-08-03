#!/usr/bin/env python3
"""Verify the staged XIH-Bench HuggingFace release before uploading anything.

Checks, in order:
  A  round-trip: materialize.py output is byte-identical to benchmark/ (774 files)
  B  row counts per config/split match the paper's Table `tab:xihbench_size`
  C  structural invariants (shard layout, safety asymmetry, label balance, ...)
  D  load_dataset() works on the staged folder, including a single-language-pair
     pull via data_files

Nothing here touches the Hub. If this passes locally it will behave the same way
once uploaded, because load_dataset reads the card's YAML identically in both
places.

Usage:
    python src/hf/verify_release.py --staging hf --benchmark benchmark
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pyarrow.parquet as pq

LANGS = ["en", "de", "hi", "zh", "es", "fr"]
DOMAINS = ["rule-following", "safety", "task-execution", "persona"]

EXPECTED_ROWS = {
    ("rule-following", "reference"): 10800, ("rule-following", "conflict"): 10800,
    ("task-execution", "reference"): 10800, ("task-execution", "conflict"): 10800,
    ("persona", "reference"): 10800, ("persona", "conflict"): 10800,
    ("safety", "reference"): 3294, ("safety", "conflict"): 10800,
    ("all", "reference"): 35694, ("all", "conflict"): 43200,
}
EXPECTED_SHARDS = {
    "all": 72, "rule-following": 72, "task-execution": 72, "persona": 72, "safety": 42,
}

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


def sha_tree(root, rel_prefix="benchmark"):
    """sha256 of every language-pair file, keyed by path relative to root."""
    out = {}
    base = os.path.join(root, rel_prefix)
    for dirpath, _, files in os.walk(base):
        for f in files:
            if not f.startswith("input-"):
                continue  # skips the 14 generation-time assets
            p = os.path.join(dirpath, f)
            out[os.path.relpath(p, root)] = hashlib.sha256(
                open(p, "rb").read()).hexdigest()
    return out


def check_roundtrip(staging, benchmark):
    print("\nA. round-trip byte equality")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, "src/hf/materialize.py",
             "--parquet", os.path.join(staging, "data", "all"), "--out", tmp],
            capture_output=True, text=True)
        if r.returncode != 0:
            check("materialize.py runs", False, r.stderr.strip()[-500:])
            return
        check("materialize.py runs", True)

        want = sha_tree(os.path.dirname(os.path.abspath(benchmark)) or ".",
                        os.path.basename(benchmark))
        got = sha_tree(tmp, os.path.basename(benchmark))

        check("774 files rebuilt", len(got) == 774, f"got {len(got)}")
        check("774 source files present", len(want) == 774, f"got {len(want)}")
        diff = {k for k in want if want[k] != got.get(k)}
        check("all rebuilt files byte-identical", not diff,
              f"{len(diff)} differ, e.g. {sorted(diff)[:3]}")


def check_counts(staging):
    print("\nB. row counts")
    grand = 0
    for (config, setting), expected in sorted(EXPECTED_ROWS.items()):
        pattern = os.path.join(staging, "data", config, f"{setting}-*.parquet")
        n = sum(pq.read_metadata(p).num_rows for p in glob.glob(pattern))
        check(f"{config}/{setting} == {expected:,}", n == expected, f"got {n:,}")
        if config != "all":
            grand += n
    check("grand total == 78,894", grand == 78894, f"got {grand:,}")


def check_structure(staging):
    print("\nC. structural invariants")
    total_shards = 0
    for config, expected in sorted(EXPECTED_SHARDS.items()):
        files = glob.glob(os.path.join(staging, "data", config, "*.parquet"))
        check(f"{config} has {expected} shards", len(files) == expected, f"got {len(files)}")
        total_shards += len(files)
    check("330 Parquet files total", total_shards == 330, f"got {total_shards}")

    # Every shard's filename must agree with the languages inside it.
    mismatched = []
    for p in glob.glob(os.path.join(staging, "data", "*", "*.parquet")):
        setting, hi, lo, _ = os.path.basename(p).split("-", 3)
        t = pq.read_table(p, columns=["setting", "higher_lang", "lower_lang"])
        if (set(t["setting"].to_pylist()) != {setting}
                or set(t["higher_lang"].to_pylist()) != {hi}
                or set(t["lower_lang"].to_pylist()) != {lo}):
            mismatched.append(os.path.basename(p))
    check("shard filenames match their contents", not mismatched,
          f"{len(mismatched)} mismatched, e.g. {mismatched[:3]}")

    # safety/reference is diagonal-only -- the asymmetry the paper documents.
    safety_ref = sorted(os.path.basename(p) for p in
                        glob.glob(os.path.join(staging, "data", "safety", "reference-*.parquet")))
    want = sorted(f"reference-{l}-{l}-00000-of-00001.parquet" for l in LANGS)
    check("safety/reference is diagonal-only (6 shards)", safety_ref == want,
          f"got {safety_ref}")

    t = pq.read_table(os.path.join(staging, "data", "all", "conflict-en-de-00000-of-00001.parquet"))
    check("one conflict shard holds 4 domains x 3 hierarchies x 100", t.num_rows == 1200,
          f"got {t.num_rows}")

    # Aggregate column-level invariants over the union config.
    tbl = pq.read_table(os.path.join(staging, "data", "all"),
                        columns=["domain", "setting", "hierarchy", "has_system",
                                 "has_tool", "source_file", "gold_json", "row_in_file"])
    df = tbl.to_pydict()

    check("52,596 tool-bearing rows", sum(df["has_tool"]) == 52596,
          f"got {sum(df['has_tool'])}")
    check("774 distinct source_file", len(set(df["source_file"])) == 774,
          f"got {len(set(df['source_file']))}")

    ut_system = [s for h, s in zip(df["hierarchy"], df["has_system"]) if h == "user-tool"]
    check("user-tool rows never carry a system key", not any(ut_system),
          f"{sum(ut_system)} rows do")

    labels = collections.Counter()
    for dom, gold in zip(df["domain"], df["gold_json"]):
        g = json.loads(gold)
        if dom == "persona":
            labels[("persona", g["label"])] += 1
        elif dom == "safety":
            labels[("safety", g["answer"]["label"])] += 1
    check("persona labels {1: 11664, 0: 9936}",
          labels[("persona", 1)] == 11664 and labels[("persona", 0)] == 9936,
          f"got 1:{labels[('persona', 1)]} 0:{labels[('persona', 0)]}")
    check("safety labels {0: 12996, 1: 1098}",
          labels[("safety", 0)] == 12996 and labels[("safety", 1)] == 1098,
          f"got 0:{labels[('safety', 0)]} 1:{labels[('safety', 1)]}")


def check_load_dataset(staging):
    print("\nD. load_dataset")
    try:
        from datasets import load_dataset
    except ImportError:
        check("datasets importable", False, "pip install datasets")
        return

    card = os.path.join(staging, "README.md")
    if not os.path.exists(card):
        check("card present (required for configs)", False, f"missing {card}")
        return

    try:
        d = load_dataset(staging, "rule-following", split="conflict")
        check("load_dataset(staging, 'rule-following', split='conflict') == 10,800",
              d.num_rows == 10800, f"got {d.num_rows}")
        cell = d.filter(lambda x: x["lang_pair"] == "en-zh")
        check("filter to lang_pair 'en-zh' == 300", cell.num_rows == 300,
              f"got {cell.num_rows}")
    except Exception as e:  # noqa: BLE001
        check("load_dataset on domain config", False, f"{type(e).__name__}: {e}")

    try:
        one = load_dataset("parquet", data_files=os.path.join(
            staging, "data", "rule-following", "conflict-en-zh-*.parquet"), split="train")
        check("single-language-pair pull via data_files == 300", one.num_rows == 300,
              f"got {one.num_rows}")
    except Exception as e:  # noqa: BLE001
        check("data_files single-pair pull", False, f"{type(e).__name__}: {e}")

    try:
        s = load_dataset(staging, "safety", split="reference")
        check("safety/reference == 3,294", s.num_rows == 3294, f"got {s.num_rows}")
        check("safety/reference is all same-language", all(s["same_language"]))
    except Exception as e:  # noqa: BLE001
        check("load_dataset on safety", False, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="hf")
    ap.add_argument("--benchmark", default="benchmark")
    ap.add_argument("--skip-roundtrip", action="store_true")
    args = ap.parse_args()

    if not args.skip_roundtrip:
        check_roundtrip(args.staging, args.benchmark)
    check_counts(args.staging)
    check_structure(args.staging)
    check_load_dataset(args.staging)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
