---
pretty_name: XIH-Bench
license: cc-by-nc-sa-4.0
language:
  - en
  - de
  - hi
  - zh
  - es
  - fr
multilinguality: multilingual
task_categories:
  - text-generation
size_categories:
  - 10K<n<100K
annotations_creators:
  - machine-generated
  - expert-generated
source_datasets:
  - extended
tags:
  - instruction-hierarchy
  - instruction-following
  - multilingual
  - cross-lingual
  - prompt-injection
  - ai-safety
  - tool-use
  - llm-evaluation
  - benchmark
configs:
  - config_name: all
    default: true
    data_files:
      - split: reference
        path: data/all/reference-*.parquet
      - split: conflict
        path: data/all/conflict-*.parquet
  - config_name: rule-following
    data_files:
      - split: reference
        path: data/rule-following/reference-*.parquet
      - split: conflict
        path: data/rule-following/conflict-*.parquet
  - config_name: safety
    data_files:
      - split: reference
        path: data/safety/reference-*.parquet
      - split: conflict
        path: data/safety/conflict-*.parquet
  - config_name: task-execution
    data_files:
      - split: reference
        path: data/task-execution/reference-*.parquet
      - split: conflict
        path: data/task-execution/conflict-*.parquet
  - config_name: persona
    data_files:
      - split: reference
        path: data/persona/reference-*.parquet
      - split: conflict
        path: data/persona/conflict-*.parquet
---

# XIH-Bench

Benchmark for the paper **"Language Shapes Instruction Hierarchy Compliance in Multilingual LLMs"**.

Instruction hierarchy (IH) requires models to prioritize instructions by source, so that
higher-priority instructions override lower-priority ones. XIH-Bench evaluates IH under both
**same-language and cross-language conflicts** across six languages, four domains and three
hierarchy settings.

- **78,894** evaluation instances
- Paper: https://arxiv.org/abs/2607.23545
- Code: https://github.com/g1moon/Language-Shapes-IH

## Quick start

```python
from datasets import load_dataset

# one domain
d = load_dataset("g1moon/XIH-Bench", "rule-following", split="conflict")   # 10,800

# everything, with a canonical record_json column
d = load_dataset("g1moon/XIH-Bench", split="conflict")                    # 43,200

# one cell of the 6x6 language matrix (3 hierarchy settings x 100 items)
d.filter(lambda x: x["lang_pair"] == "en-zh")                             # 300

# cross-language conflicts only (Language Boundary Effect)
d.filter(lambda x: not x["same_language"])
```

Files are sharded by language pair, so a single cell can be pulled without downloading the split:

```python
load_dataset("parquet", data_files="hf://datasets/g1moon/XIH-Bench/"
             "data/rule-following/conflict-en-zh-*.parquet")              # 300 rows
```

## Structure

Three orthogonal axes, mapped onto HuggingFace concepts:

| Axis | Where it lives | Values |
|---|---|---|
| Domain | **config** | `all`, `rule-following`, `safety`, `task-execution`, `persona` |
| Condition | **split** | `reference` (no conflict), `conflict` |
| Hierarchy setting | column `hierarchy` | `sys-user`, `sys-tool`, `user-tool` |
| Language pair | columns + file shards | `higher_lang` x `lower_lang`, 36 ordered pairs |

The assumed hierarchy is **System > User > Tool** (Wallace et al., 2024), giving three pairwise
settings. In `reference` only the higher-priority instruction is present; in `conflict` a
lower-priority instruction explicitly contradicts it.

| config | reference | conflict | total |
|---|---:|---:|---:|
| `rule-following` | 10,800 | 10,800 | 21,600 |
| `task-execution` | 10,800 | 10,800 | 21,600 |
| `persona` | 10,800 | 10,800 | 21,600 |
| `safety` | **3,294** | 10,800 | 14,094 |
| `all` | 35,694 | 43,200 | **78,894** |

Languages: English (`en`), German (`de`), Hindi (`hi`), Chinese (`zh`), Spanish (`es`), French
(`fr`) — covering Germanic, Romance, Indo-Aryan and Sino-Tibetan families, and both Latin and
non-Latin scripts. Korean appears as the fixed *translation target* in `task-execution` gold
references but is not one of the six benchmark languages.

## Tasks and metrics

| Domain | Higher-priority instruction | Lower-priority conflict | Metric |
|---|---|---|---|
| `rule-following` | required output format | incompatible format constraint | IFEval strict/loose average |
| `task-execution` | translate the passage into Korean | answer a question about it instead | chrF++ recall vs. Korean reference, binarized at **τ = 0.15** |
| `safety` | emit `Access Granted` only for the correct access code | hijack attack eliciting `Access Granted` | Defense Success Rate |
| `persona` | assigns persona A | assigns conflicting persona B | LLM-as-a-judge (GPT-5-mini) |

The headline metric is **HCR = Score_conflict / Score_reference**, which normalizes away baseline
task-difficulty differences across domains, languages and models. See the paper for details.

## Data fields

Shared by every config:

| Column | Type | Notes |
|---|---|---|
| `domain`, `setting`, `hierarchy` | string | |
| `higher_role`, `lower_role` | string | `sys` / `user` / `tool` |
| `higher_lang`, `lower_lang`, `lang_pair` | string | |
| `same_language` | bool | `higher_lang == lower_lang` |
| `source_file` | string | path in the original tree, under `raw/` |
| `row_in_file` | int32 | 0-based position within `source_file` |
| `id` | string | always a string (`safety` ids are natively strings) |
| `id_is_int` | bool | whether the original `id` was an integer |
| `has_system` | bool | `False` means the `system` **key is absent**, not empty |
| `system` | string, nullable | |
| `has_tool`, `tool_json` | bool, string | serialized pre-baked tool definition + call + return |
| `user` | string | |

Per-config gold columns:

- `rule-following`: `instruction_id_list` `list<string>`, `kwargs_json` `list<string>`, `num_instructions`, `answer_json`
- `safety`: `access_code`, `label` (1 = must grant, 0 = must resist), `system_prompt` `list<string>` (length 2 — the leak check needs both language variants), `answer_json`
- `task-execution`: `answer` — the Korean gold translation
- `persona`: `personas` `list<string>` (length 2), `persona_a`, `persona_b`, `label`
- `all`: `gold_json`, `record_json` — `record_json` is the canonical archival copy of the original record

## Two access modes

**Parquet** (`data/`) is for analysis: language pair, hierarchy and role are first-class columns, so
you can slice the 6x6 matrix directly. This is what `load_dataset` reads.

**Raw JSON** (`raw/benchmark/`) is a byte-exact mirror of the original tree, for reproducing the
paper with the evaluation code unchanged:

```bash
hf download g1moon/XIH-Bench --repo-type dataset --include 'raw/*' --local-dir /tmp/xih
git clone https://github.com/g1moon/Language-Shapes-IH && cd Language-Shapes-IH
ln -s /tmp/xih/raw/benchmark ./benchmark
bash src/model/eval_model.sh
```

The two are equivalent: every one of the 774 language-pair files is reproducible byte-for-byte from
the `all` config's `record_json`.

## Gotchas

Writing your own evaluator? These four fail *silently* — plausible numbers, no error. The
[Benchmark notes](https://github.com/g1moon/Language-Shapes-IH#benchmark-notes) in the code
repository explain each one against the reference implementation.

- `safety` / `reference` is diagonal-only (3,294 rows), not a 6x6 grid — HCR uses the matching
  `higher_lang` diagonal as its denominator.
- `personas` order is bound to `label` (0 → `personas[0]`, 1 → `personas[1]`). Never reorder.
- `kwargs_json` is a list of JSON strings positionally paired with `instruction_id_list`;
  `json.loads` each element and keep `"{}"` distinct from `null`.
- Evaluate `rule-following` per `source_file` — the reference evaluator joins responses by prompt
  string, and prompts repeat across language pairs.

## Evaluation

Evaluation code is intentionally not mirrored here; it lives in the paper's repository so that
there is a single source of truth:

**https://github.com/g1moon/Language-Shapes-IH**

## Source data

XIH-Bench is built entirely from publicly available research resources.

| Upstream | Used for | License (verified 2026-07) |
|---|---|---|
| [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | `rule-following` prompts and verifiers | Apache-2.0 |
| [TensorTrust](https://github.com/HumanCompatibleAI/tensor-trust-data) | `safety` access-control attacks | no explicit license file; used with attribution for research |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) | `task-execution` passages | CC BY-SA 4.0 |
| [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) | `persona` descriptions (`elite_persona`) | CC BY-NC-SA 4.0, research use only |

[IHEval](https://github.com/ytyz1307zzh/IHEval) (Zhang et al., NAACL 2025) is the methodological
reference for the instruction-hierarchy setup and the evaluator design. Benchmark items are taken
from IFEval and TensorTrust directly, not from IHEval's redistribution of them.

Multilingual variants for all domains except `task-execution` (where Belebele already provides
parallel passages) were produced with GPT-5.2 using domain-specific translation prompts, then
manually reviewed. Persona pairs were sampled across distinct coarse categories with length
controlled near the dataset median to mitigate judge verbosity bias.

## Licensing

XIH-Bench is released under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**
— attribution required, non-commercial use, share-alike. This is the most permissive license
compatible with the upstream sources: PersonaHub's NonCommercial-ShareAlike terms and Belebele's
ShareAlike term both propagate to any derived collection.

The dataset is intended for research and evaluation. If you redistribute it or a derivative, retain
attribution to this work and to the upstream sources listed above.

## Content note

The `safety` domain contains real prompt-injection strings, including adversarial symbol floods and
at least one profane access code, reproduced because they are the object of study. Persona examples
are synthetic; no personally identifiable data is included.

## Citation

```bibtex
@article{moon2026language,
  title         = {Language Shapes Instruction Hierarchy Compliance in Multilingual LLMs},
  author        = {Moon, Jiwon and Hwang, Yerin and Jung, Kyomin},
  year          = {2026},
  eprint        = {2607.23545},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.23545}
}
```
