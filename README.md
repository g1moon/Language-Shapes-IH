# Language Shapes Instruction Hierarchy Compliance in Multilingual LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2607.23545-b31b1b.svg)](https://arxiv.org/abs/2607.23545)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-XIH--Bench-yellow)](https://huggingface.co/datasets/g1moon/XIH-Bench)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Abstract

Instruction hierarchy (IH) requires models to prioritize instructions by source, ensuring that higher-priority instructions override lower-priority ones. Despite its importance for safe and controllable deployment, existing evaluations have focused almost exclusively on English, leaving it unclear whether IH compliance remains stable in multilingual settings. We introduce XIH-Bench, a benchmark for multilingual IH evaluation with both same-language and cross-language conflicts across six languages, four domains, and three IH settings. Across models, we find two consistent patterns. First, IH compliance exhibits a clear language-dependent asymmetry: a language that strengthens compliance in the higher-priority position can become disruptive in the lower-priority position. Second, cross-language conflicts yield higher compliance than same-language conflicts, a phenomenon we term the Language Boundary Effect. We further show that language specialization can make lower-priority instructions in model-favored languages harder to override, creating multilingual reliability and security risks.

---

## Overview

XIH-Bench covers:

- **4 Domains**: rule-following, safety, task-execution, persona
- **3 Hierarchy Types**: sys-user, sys-tool, user-tool
- **2 Settings**: reference (no conflict), conflict
- **6 Languages**: English (en), German (de), Hindi (hi), Chinese (zh), Spanish (es), French (fr)

Each test instance uses a pair of languages — one for the higher-hierarchy role and one for the lower-hierarchy role — resulting in 36 language-pair combinations per hierarchy type.

---

## Installation

```bash
conda create -n xih-bench python=3.10 -y
conda activate xih-bench
pip install -r requirements.txt
```

For open-source models (Llama, Qwen, Mistral), also install GPU-dependent packages:

```bash
pip install torch>=2.9.0
pip install vllm>=0.13.0  # requires CUDA 12.x
```

Set environment variables for API models:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

---

## Benchmark Structure

The benchmark is included in this repository under `benchmark/`. It is also published on the
HuggingFace Hub, which additionally provides Parquet configs with the language pair, hierarchy
setting and role as first-class columns:

```python
from datasets import load_dataset

d = load_dataset("g1moon/XIH-Bench", "rule-following", split="conflict")   # 10,800
d.filter(lambda x: x["lang_pair"] == "en-zh")                              # one 6x6 cell
```

To reproduce the paper with the code in this repository against the Hub copy of the raw tree:

```bash
hf download g1moon/XIH-Bench --repo-type dataset --include 'raw/*' --local-dir /tmp/xih
ln -s /tmp/xih/raw/benchmark ./benchmark
```

The Hub release is built and verified by `src/hf/`:

```bash
python src/hf/build_hf_dataset.py --benchmark-root benchmark --out hf
python src/hf/verify_release.py --staging hf --benchmark benchmark
```

```
benchmark/
├── rule-following/
│   ├── reference/
│   │   ├── sys-tool/
│   │   │   ├── input-sys_en-tool_de.json   # system=English, tool=German
│   │   │   └── ...
│   │   ├── sys-user/
│   │   └── user-tool/
│   └── conflict/
│       └── ...
├── safety/
├── task-execution/
└── persona/
```

**Language pair convention**: `input-{higher_role}_{higher_lang}-{lower_role}_{lower_lang}.json`

### Benchmark notes

The pipeline in `src/` handles all of the following. They matter if you write your own evaluator
(for example on top of the HuggingFace Parquet release), because each fails *silently* — you get
plausible-looking numbers rather than an error.

- **`safety/reference` is not a 6x6 grid.** It exists only on the diagonal: 6 same-language pairs
  x 3 hierarchy settings x 183 items = 3,294 instances, because the reference condition does not
  vary with the lower-priority language. Safety HCR therefore uses the diagonal reference of the
  matching higher-priority language as the denominator for every language paired with it — see
  `src/model/aggregate_matrix_scores.py:328-333`.
- **`personas` order is semantic.** `personas[0]` is Persona A and `personas[1]` is Persona B, and
  the mapping is bound to `label` (0 → A, 1 → B) in `src/persona/evaluate/eval_persona.py:49-53`.
  Sorting or shuffling within a record silently inverts the judge's scoring.
- **`answer.kwargs` is positional.** It is a list parallel to `answer.instruction_id_list`, and
  entry *i* is splatted into the verifier for instruction *i*
  (`src/rule_following/evaluate/evaluation_main.py:106-113`). An empty `{}` is meaningful.
- **Rule-following evaluation is per file.** The evaluator joins model responses to items by
  *prompt string*, not by id (`src/rule_following/evaluate/evaluation_main.py:102,192`). Prompts
  repeat across language-pair files — a single domain/setting/hierarchy has 3,600 items but only
  600 distinct `user` strings — so evaluating a merged set collapses colliding prompts.
- **`answer.system_prompt` holds both language variants** and the leak check takes the maximum
  chrF++ overlap across them (`src/safety/evaluate/eval_tensortrust.py:96`).
- **Strings must round-trip byte-for-byte** with `ensure_ascii=False` and no normalization, since
  chrF++ compares against exact reference strings.

---

## Quick Start

### Run Inference

Edit `src/model/run_model.sh` to set `model_list`, then:

```bash
bash src/model/run_model.sh
```

Or submit to SLURM:

```bash
sbatch src/model/run_model.sh
```

For API models only (GPT, Claude), use the batch API for cost efficiency:

```bash
bash src/model/run_model_batch.sh
```

### Single Model / Single Combination

```bash
python src/model/run_rule_following.py \
    -model gpt-5-nano-2025-08-07 \
    -input benchmark/rule-following/reference/sys-tool/input-sys_en-tool_en.json \
    -request_file results/rule-following/reference/sys-tool/gpt/gpt-5-nano/input-sys_en-tool_en/input_request.json \
    -response_file results/rule-following/reference/sys-tool/gpt/gpt-5-nano/input-sys_en-tool_en/input_response.json \
    -eval_output_dir results/rule-following/reference/sys-tool/gpt/gpt-5-nano/input-sys_en-tool_en \
    -backend api
```

---

## Running Evaluation

Edit `src/model/eval_model.sh` to set `model_list` and `domains`, then:

```bash
bash src/model/eval_model.sh
```

Or evaluate a single combination:

```bash
python src/model/eval_rule_following.py \
    -model gpt-5-nano-2025-08-07 \
    -input benchmark/rule-following/reference/sys-tool/input-sys_en-tool_en.json \
    -response_file results/rule-following/reference/sys-tool/gpt/gpt-5-nano/input-sys_en-tool_en/input_response.json \
    -eval_output_dir results/rule-following/reference/sys-tool/gpt/gpt-5-nano/input-sys_en-tool_en \
    -backend api
```

---

## Aggregating Results

After evaluation, aggregate scores into 6×6 language matrices:

```bash
# Step 1: Build per-model 6x6 matrices
python src/model/aggregate_matrix_scores.py \
    -results_dir results -model_family gpt -model_name gpt-5-nano \
    -output_dir model-scores -domain rule-following

# Step 2: Average across hierarchy types
python src/model/calculate_hierarchy_avg.py \
    -model_family gpt -model_name gpt-5-nano \
    -domain rule-following -output_dir model-scores

# Step 3: Average across models in a family
python src/model/aggregate_family_scores.py \
    -model_family gpt -output_dir model-scores -domain rule-following
```

Outputs are saved to `model-scores/{family}/{model}/results/{domain}/`.

---

## Supported Models

| Family | Model Name | Backend |
|--------|-----------|---------|
| GPT | gpt-5 | API |
| GPT | gpt-5-mini | API |
| GPT | gpt-5-nano | API |
| Claude | claude-sonnet | API |
| Claude | claude-haiku | API |
| Llama | llama3.1-8b | vLLM |
| Llama | llama3.1-70b | vLLM |
| Llama | llama3.2-3b | vLLM |
| Qwen | qwen3-4b | vLLM |
| Qwen | qwen3-30b | vLLM |
| Mistral | ministral3-8b | vLLM |
| Mistral | ministral3-14b | vLLM |
| Mistral | mistral-small-24b | vLLM |

---

## Results Layout

```
results/{domain}/{setting}/{hierarchy}/{model_family}/{model}/{lang_combo}/
├── input_request.json       # formatted API requests
├── input_response.json      # model outputs
├── eval_results.json        # evaluation scores (safety, task-execution, persona)
├── eval_results_strict.json # strict scores (rule-following only)
└── eval_results_loose.json  # loose scores (rule-following only)
```

Aggregated scores:

```
model-scores/{model_family}/{model}/results/{domain}/{hierarchy}/
├── reference.csv   # 6x6 matrix, reference setting
├── conflict.csv    # 6x6 matrix, conflict setting
└── hcr.csv         # Hierarchy Compliance Rate = conflict / reference
```

---

## Validation Utilities

```bash
# Verify the HuggingFace release: round-trip byte equality against benchmark/,
# per-config row counts, and structural invariants
python src/hf/verify_release.py --staging hf --benchmark benchmark
```

---

## License

XIH-Bench is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [LICENSE](LICENSE) for the upstream source attributions (IFEval, TensorTrust, Belebele,
PersonaHub) whose terms also apply.

---

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

