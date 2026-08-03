#!/usr/bin/env python3
"""Rebuild the original benchmark/ JSON tree from the `all` Parquet config.

This exists to prove the Parquet release is lossless -- every one of the 774
language-pair files comes back byte-for-byte. It is a build-time verification
tool and is deliberately NOT shipped to the Hub: the Hub release already carries
the original tree under raw/, so nobody downstream needs to reconstruct it.

The 14 generation-time assets (input_{lang}.json, tool_template.json) are not
represented in Parquet and are not regenerated here; raw/ is their only home.

Usage:
    python src/hf/materialize.py --parquet hf/data/all --out /tmp/xih-rebuild
    python src/hf/materialize.py --repo g1moon/XIH-Bench --out /tmp/xih-rebuild
"""

import argparse
import glob
import json
import os
import sys

import pyarrow.parquet as pq


def load_rows(parquet_dir):
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    assert files, f"no Parquet files under {parquet_dir}"
    rows = []
    for path in files:
        t = pq.read_table(path, columns=["source_file", "row_in_file", "record_json"])
        rows.extend(zip(t["source_file"].to_pylist(),
                        t["row_in_file"].to_pylist(),
                        t["record_json"].to_pylist()))
    return files, rows


def rebuild(rows, out_root):
    """Group by source_file and write each file in its original order."""
    groups = {}
    for source_file, row_in_file, record_json in rows:
        groups.setdefault(source_file, []).append((row_in_file, record_json))

    for source_file, entries in sorted(groups.items()):
        entries.sort(key=lambda e: e[0])
        assert [i for i, _ in entries] == list(range(len(entries))), \
            f"{source_file}: row_in_file is not a contiguous 0..n-1 range"

        records = [json.loads(rj) for _, rj in entries]
        # 774/774 tree files are byte-exactly this serialization, no trailing
        # newline. Key order survives because json.loads preserves it.
        text = json.dumps(records, indent=2, ensure_ascii=False)

        dest = os.path.join(out_root, source_file)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(text)

    return len(groups)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet", help="local directory holding the `all` config Parquet")
    src.add_argument("--repo", help="Hub dataset id, e.g. g1moon/XIH-Bench")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parquet_dir = args.parquet
    if args.repo:
        from huggingface_hub import snapshot_download
        local = snapshot_download(args.repo, repo_type="dataset",
                                  allow_patterns="data/all/*")
        parquet_dir = os.path.join(local, "data", "all")

    files, rows = load_rows(parquet_dir)
    print(f"read {len(rows):,} rows from {len(files)} Parquet files")

    n = rebuild(rows, args.out)
    print(f"wrote {n} JSON files under {args.out}")
    assert n == 774, f"expected 774 files, wrote {n}"


if __name__ == "__main__":
    sys.exit(main())
