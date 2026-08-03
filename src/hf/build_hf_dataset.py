#!/usr/bin/env python3
"""Build the HuggingFace Hub release of XIH-Bench from the benchmark/ tree.

Emits into --out:
    data/{config}/{split}-{higher_lang}-{lower_lang}-00000-of-00001.parquet
    raw/benchmark                (symlink by default, real copy with --copy-raw)

Configs are one per domain plus a union config `all` that carries `record_json`,
the canonical column materialize.py reads to rebuild the original JSON tree.

Nothing is coerced silently: every structural expectation in the benchmark is
asserted here, so a schema change upstream fails the build instead of shipping
malformed Parquet.

Usage:
    python src/hf/build_hf_dataset.py --benchmark-root benchmark --out hf
"""

import argparse
import json
import os
import re
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

DOMAINS = ["rule-following", "safety", "task-execution", "persona"]
SETTINGS = ["reference", "conflict"]
HIERARCHIES = ["sys-user", "sys-tool", "user-tool"]
LANGS = ["en", "de", "hi", "zh", "es", "fr"]

# input-{higher_role}_{higher_lang}-{lower_role}_{lower_lang}.json
FNAME_RE = re.compile(r"^input-(sys|user)_([a-z]{2})-(user|tool)_([a-z]{2})\.json$")

# Generation-time assets that no runtime code reads. They ship verbatim under
# raw/ and are deliberately absent from the Parquet configs.
EXTRA_ASSET_RE = re.compile(r"^(input_[a-z]{2}|tool_template)\.json$")

COMMON_FIELDS = [
    ("domain", pa.string()),
    ("setting", pa.string()),
    ("hierarchy", pa.string()),
    ("higher_role", pa.string()),
    ("lower_role", pa.string()),
    ("higher_lang", pa.string()),
    ("lower_lang", pa.string()),
    ("lang_pair", pa.string()),
    ("same_language", pa.bool_()),
    ("source_file", pa.string()),
    ("row_in_file", pa.int32()),
    ("id", pa.string()),
    ("id_is_int", pa.bool_()),
    ("has_system", pa.bool_()),
    ("system", pa.string()),
    ("has_tool", pa.bool_()),
    ("tool_json", pa.string()),
    ("user", pa.string()),
]

# kwargs_json is list<string> rather than a struct on purpose: the IFEval
# verifier does build_description(**kwargs[i]), so "key absent" and "key present
# but null" are different things a nullable struct would conflate.
EXTRA_FIELDS = {
    "rule-following": [
        ("instruction_id_list", pa.list_(pa.string())),
        ("kwargs_json", pa.list_(pa.string())),
        ("num_instructions", pa.int32()),
        ("answer_json", pa.string()),
    ],
    "safety": [
        ("access_code", pa.string()),
        ("label", pa.int32()),
        ("system_prompt", pa.list_(pa.string())),
        ("answer_json", pa.string()),
    ],
    "task-execution": [
        ("answer", pa.string()),
    ],
    "persona": [
        ("personas", pa.list_(pa.string())),
        ("persona_a", pa.string()),
        ("persona_b", pa.string()),
        ("label", pa.int32()),
    ],
    "all": [
        ("gold_json", pa.string()),
        ("record_json", pa.string()),
    ],
}

SCHEMAS = {
    cfg: pa.schema(COMMON_FIELDS + extra) for cfg, extra in EXTRA_FIELDS.items()
}


def dumps(obj):
    """Serialize for the JSON-string columns. Never escape non-ASCII."""
    return json.dumps(obj, ensure_ascii=False)


def common_row(rec, meta, row_in_file):
    """The columns shared by every config."""
    has_system = "system" in rec
    has_tool = "tool" in rec
    return {
        "domain": meta["domain"],
        "setting": meta["setting"],
        "hierarchy": meta["hierarchy"],
        "higher_role": meta["higher_role"],
        "lower_role": meta["lower_role"],
        "higher_lang": meta["higher_lang"],
        "lower_lang": meta["lower_lang"],
        "lang_pair": meta["lang_pair"],
        "same_language": meta["higher_lang"] == meta["lower_lang"],
        "source_file": meta["source_file"],
        "row_in_file": row_in_file,
        "id": str(rec["id"]),
        "id_is_int": isinstance(rec["id"], int),
        "has_system": has_system,
        "system": rec["system"] if has_system else None,
        "has_tool": has_tool,
        "tool_json": dumps(rec["tool"]) if has_tool else None,
        "user": rec["user"],
    }


def domain_extras(domain, rec, where):
    """Per-domain gold columns, plus the assertions that guard them."""
    if domain == "rule-following":
        ans = rec["answer"]
        ids, kwargs = ans["instruction_id_list"], ans["kwargs"]
        assert len(ids) == len(kwargs), f"{where}: {len(ids)} ids vs {len(kwargs)} kwargs"
        return {
            "instruction_id_list": list(ids),
            "kwargs_json": [dumps(k) for k in kwargs],
            "num_instructions": len(ids),
            "answer_json": dumps(ans),
        }

    if domain == "safety":
        ans = rec["answer"]
        assert isinstance(rec["id"], str), f"{where}: safety id must be str, got {type(rec['id'])}"
        assert len(ans["system_prompt"]) == 2, f"{where}: system_prompt len {len(ans['system_prompt'])}"
        assert ans["label"] in (0, 1), f"{where}: label {ans['label']}"
        return {
            "access_code": ans["access_code"],
            "label": ans["label"],
            "system_prompt": list(ans["system_prompt"]),
            "answer_json": dumps(ans),
        }

    if domain == "task-execution":
        assert isinstance(rec["answer"], str), f"{where}: answer must be str"
        return {"answer": rec["answer"]}

    if domain == "persona":
        assert "answer" not in rec, f"{where}: persona must not carry an 'answer' key"
        personas = rec["personas"]
        assert len(personas) == 2, f"{where}: personas len {len(personas)}"
        assert rec["label"] in (0, 1), f"{where}: label {rec['label']}"
        # personas[0] is Persona A and personas[1] Persona B; the order is bound
        # to `label` by the judge (src/persona/evaluate/eval_persona.py:49-53).
        return {
            "personas": list(personas),
            "persona_a": personas[0],
            "persona_b": personas[1],
            "label": rec["label"],
        }

    raise AssertionError(f"unknown domain {domain}")


def all_extras(domain, rec):
    """Union-config columns. record_json is the canonical archival copy."""
    if domain == "persona":
        gold = {"personas": rec["personas"], "label": rec["label"]}
    else:
        gold = {"answer": rec["answer"]}
    return {"gold_json": dumps(gold), "record_json": dumps(rec)}


def collect(benchmark_root):
    """Walk the tree, returning {(config, split, hi_lang, lo_lang): [rows]}."""
    buckets = {}
    n_matched = n_skipped = 0

    for domain in DOMAINS:
        for setting in SETTINGS:
            for hierarchy in HIERARCHIES:
                d = os.path.join(benchmark_root, domain, setting, hierarchy)
                assert os.path.isdir(d), f"missing directory {d}"
                for fname in sorted(os.listdir(d)):
                    m = FNAME_RE.match(fname)
                    if not m:
                        raise AssertionError(f"unexpected file in tree: {d}/{fname}")
                    hi_role, hi_lang, lo_role, lo_lang = m.groups()
                    assert f"{hi_role}-{lo_role}" == hierarchy, \
                        f"{d}/{fname}: roles {hi_role}-{lo_role} != dir {hierarchy}"
                    assert hi_lang in LANGS and lo_lang in LANGS, f"{d}/{fname}: bad language"

                    path = os.path.join(d, fname)
                    source_file = os.path.relpath(path, os.path.dirname(benchmark_root) or ".")
                    meta = {
                        "domain": domain, "setting": setting, "hierarchy": hierarchy,
                        "higher_role": hi_role, "lower_role": lo_role,
                        "higher_lang": hi_lang, "lower_lang": lo_lang,
                        "lang_pair": f"{hi_lang}-{lo_lang}",
                        "source_file": source_file,
                    }

                    with open(path, encoding="utf-8") as fh:
                        records = json.load(fh)
                    assert isinstance(records, list), f"{path}: expected a JSON array"

                    seen_ids = set()
                    for i, rec in enumerate(records):
                        where = f"{source_file}[{i}]"
                        assert str(rec["id"]) not in seen_ids, f"{where}: duplicate id"
                        seen_ids.add(str(rec["id"]))
                        if "tool" in rec:
                            assert set(rec["tool"]) == {"definition", "call", "return"}, \
                                f"{where}: unexpected tool keys {sorted(rec['tool'])}"

                        base = common_row(rec, meta, i)
                        dom_row = dict(base, **domain_extras(domain, rec, where))
                        all_row = dict(base, **all_extras(domain, rec))

                        # Self-check: record_json must reproduce the on-disk form.
                        assert json.loads(all_row["record_json"]) == rec, f"{where}: record_json drift"

                        key = (setting, hi_lang, lo_lang)
                        buckets.setdefault((domain,) + key, []).append(dom_row)
                        buckets.setdefault(("all",) + key, []).append(all_row)

                    n_matched += 1

    # Confirm the 14 generation-time assets are where we expect and untouched.
    for domain in ("persona", "rule-following"):
        for fname in sorted(os.listdir(os.path.join(benchmark_root, domain))):
            if os.path.isdir(os.path.join(benchmark_root, domain, fname)):
                continue
            assert EXTRA_ASSET_RE.match(fname), f"unexpected asset {domain}/{fname}"
            n_skipped += 1

    assert n_matched == 774, f"expected 774 tree files, matched {n_matched}"
    assert n_skipped == 14, f"expected 14 generation-time assets, found {n_skipped}"
    return buckets


def write_parquet(buckets, out_dir):
    """One Parquet file per (config, split, language pair)."""
    counts = {}
    for (config, setting, hi, lo), rows in sorted(buckets.items()):
        cfg_dir = os.path.join(out_dir, "data", config)
        os.makedirs(cfg_dir, exist_ok=True)
        path = os.path.join(cfg_dir, f"{setting}-{hi}-{lo}-00000-of-00001.parquet")

        schema = SCHEMAS[config]
        table = pa.Table.from_pydict(
            {f.name: [r[f.name] for r in rows] for f in schema},
            schema=schema,
        )
        pq.write_table(table, path, compression="zstd", compression_level=9,
                       use_dictionary=True)
        counts[(config, setting)] = counts.get((config, setting), 0) + len(rows)
    return counts


def stage_raw(benchmark_root, out_dir, copy_raw):
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, "benchmark")
    if os.path.islink(dest) or os.path.exists(dest):
        (shutil.rmtree if os.path.isdir(dest) and not os.path.islink(dest) else os.unlink)(dest)
    if copy_raw:
        shutil.copytree(benchmark_root, dest)
        return f"copied {dest}"
    os.symlink(os.path.relpath(os.path.abspath(benchmark_root), raw_dir), dest)
    return f"symlinked {dest} -> {os.readlink(dest)}"


def emit_yaml(counts):
    """Print the card's configs: block with the measured row counts."""
    lines = ["configs:"]
    for config in ["all"] + DOMAINS:
        lines.append(f"  - config_name: {config}")
        if config == "all":
            lines.append("    default: true")
        lines.append("    data_files:")
        for setting in SETTINGS:
            n = counts[(config, setting)]
            lines.append(f"      - split: {setting}   # {n:,} rows")
            lines.append(f"        path: data/{config}/{setting}-*.parquet")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="benchmark")
    ap.add_argument("--out", default="hf")
    ap.add_argument("--copy-raw", action="store_true",
                    help="copy benchmark/ into raw/ instead of symlinking it")
    args = ap.parse_args()

    print(f"reading {args.benchmark_root} ...", flush=True)
    buckets = collect(args.benchmark_root)

    print(f"writing Parquet to {args.out}/data ...", flush=True)
    counts = write_parquet(buckets, args.out)

    n_files = sum(len(fs) for _, _, fs in os.walk(os.path.join(args.out, "data")))
    assert n_files == 330, f"expected 330 Parquet files, wrote {n_files}"
    assert len(buckets) == 330, f"expected 330 buckets, got {len(buckets)}"

    print(stage_raw(args.benchmark_root, args.out, args.copy_raw), flush=True)

    print(f"\n{n_files} Parquet files\n")
    grand = 0
    for config in ["all"] + DOMAINS:
        ref, conf = counts[(config, "reference")], counts[(config, "conflict")]
        print(f"  {config:16s} reference {ref:>6,}  conflict {conf:>6,}  total {ref + conf:>6,}")
        if config != "all":
            grand += ref + conf
    print(f"\n  grand total (excluding the 'all' union) {grand:,}")
    assert grand == 78894, f"expected 78,894 instances, got {grand:,}"
    assert counts[("all", "reference")] + counts[("all", "conflict")] == 78894

    print("\n--- paste into the card's YAML frontmatter ---")
    print(emit_yaml(counts))


if __name__ == "__main__":
    sys.exit(main())
