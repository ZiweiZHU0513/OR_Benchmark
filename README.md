# OR Benchmark

## Purpose

This repository evaluates how well large language models can translate natural-language operations research problems into Gurobi code. The benchmark is split into two complementary evaluation modes:

1. Solver evaluation: whether the generated Gurobi code can actually run and produce the same optimal value or solver status as the ground truth.
2. Orgeval evaluation: instead of requiring the model to solve the problem directly, the generated code is exported to LP format and compared against the reference LP using structural equivalence checks.

This split is useful for two reasons:

1. It separates code executability from modeling correctness.
2. It helps identify whether failures on simple or realistic instances come from solving difficulty, data loading, or incorrect model structure.

## Datasets and Evaluation Modes

The current benchmark suite covers five targets:

1. NL4OPT: solver evaluation
2. OptiBench: solver evaluation
3. MIPLIB-NL: solver evaluation
4. MIPLIB-NL exclude failure: orgeval evaluation
5. Bench4Opt: orgeval evaluation

In practice:

1. Solver evaluation focuses on accuracy and code compile/execution success.
2. Orgeval evaluation focuses on LP build success, graph-isomorphism correctness, and finer-grained error-type distributions.

## References

If you use results derived from these datasets, cite the corresponding papers below.

1. NL4OPT: Rindranirina Ramamonjison, Timothy T. Yu, Raymond Li, Haley Li, Giuseppe Carenini, et al. (2023). [NL4Opt Competition: Formulating Optimization Problems Based on Their Natural Language Descriptions](http://arxiv.org/abs/2303.08233).
2. OptiBench: Zhicheng Yang, Yiwei Wang, Yinya Huang, Zhijiang Guo, Wei Shi, et al. (2024). [OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling](http://arxiv.org/abs/2407.09887).
3. MIPLIB-NL: Zhong Li, Hongliang Lu, Tao Wei, Wenyu Liu, Yuxuan Chen, et al. (2026). [Constructing Industrial-Scale Optimization Modeling Benchmark](http://arxiv.org/abs/2602.10450).
4. MIPLIB-NL exclude failure: We excluded samples that not support ORGEval evaluation.
5. Bench4Opt: Zhuohan Wang, Ziwei Zhu, Ziniu Li, Congliang Chen, Yu Han, et al. (2025). [ORGEval: Graph-Theoretic Evaluation of LLMs in Optimization Modeling](http://arxiv.org/abs/2510.27610).
6. The graph-theoretic structural evaluation used for the orgeval pipeline is closely related to Zhuohan Wang, Ziwei Zhu, Ziniu Li, Congliang Chen, Yu Han, et al. (2025). [ORGEval: Graph-Theoretic Evaluation of LLMs in Optimization Modeling](http://arxiv.org/abs/2510.27610).

## Data Download

The full [data](data) directory is distributed through GitHub Releases rather than tracked directly in git.

You can download the packaged dataset from the release page [data-20260526](https://github.com/ZiweiZHU0513/OR_Benchmark/releases/tag/data-20260526).

If you start from a fresh clone, run the following commands at the repository root:

```sh
gh release download data-20260526 \
  --pattern "or_benchmark_data_20260526.tar.gz" \
  --pattern "or_benchmark_data_20260526.tar.gz.sha256"
shasum -a 256 -c or_benchmark_data_20260526.tar.gz.sha256
tar -xzf or_benchmark_data_20260526.tar.gz
```

After extraction, your repository will contain the same top-level data layout expected by the evaluation scripts:

```text
data/
  NL4OPT/
  bench4opt/
  miplib-nl/
  miplib-nl_exclude_failure/
  optibench/
```

The archive already contains the top-level `data/` directory, so you should extract it at the repository root rather than inside an existing `data/` subdirectory.

## Recommended Entry Point

If you want a single script that runs the entire suite, use:

[scripts/test/test_eval_suite.sh](scripts/test/test_eval_suite.sh)

This script provides two presets:

1. minimal: a fast smoke test that runs all targets on a very small slice
2. full: a full run over all targets without slicing limits

### Minimal

The `minimal` profile runs:

1. NL4OPT / OptiBench / MIPLIB-NL / MIPLIB-NL-exclude-failure with `start=0, end=2`
2. Bench4Opt with `bench4opt_max_samples=2`

It is useful for checking:

1. Whether API calls work correctly
2. Whether the environment dependencies are available
3. Whether all five targets can run end to end
4. Whether the summary can be generated successfully

Example:

```sh
OPENAI_API_KEY=your_key OPENAI_BASE_URL=your_url \
sh scripts/test/test_eval_suite.sh --profile minimal
```

If the dependencies are not installed in the current environment, you can explicitly select the environment defined by [environment.yaml](environment.yaml):

```sh
OPENAI_API_KEY=your_key OPENAI_BASE_URL=your_url \
sh scripts/test/test_eval_suite.sh --profile minimal --conda_env vllm
```

### Full

The `full` profile runs the same five targets without slice limits.

Example:

```sh
OPENAI_API_KEY=your_key OPENAI_BASE_URL=your_url \
sh scripts/test/test_eval_suite.sh --profile full --models deepseek-v3-0324 deepseek-v4-flash
```

## Low-Level Entry Point

If you need more fine-grained control, use:

[scripts/run_eval.sh](scripts/run_eval.sh)

This is the lower-level orchestration script. It lets you control:

1. The subset of targets to run
2. The list of models
3. `start` and `end`
4. `bench4opt_max_samples`
5. The summary output location

This entry point is better suited for debugging and batch experiments. For routine use, [scripts/test/test_eval_suite.sh](scripts/test/test_eval_suite.sh) is the recommended choice.

## Outputs

Raw results are written by default to:

1. [results/nl4opt](results/nl4opt)
2. [results/optibench](results/optibench)
3. [results/miplib-nl](results/miplib-nl)
4. [results/bench4opt](results/bench4opt)

Summaries are written by default to:

1. [results/summary](results/summary)

Each summary run generates two files:

1. `summary.md` for direct inspection
2. `summary.json` for downstream analysis

The summary includes:

1. Solver evaluation result tables
2. Orgeval evaluation result tables
3. Orgeval error analysis tables

## Environment Notes

There are two practical environment notes for the current evaluation pipeline:

1. The recommended environment definition for this repository is [environment.yaml](environment.yaml). A typical setup is:

```sh
conda env create -f environment.yaml
conda activate benchmarkOR
```

If the environment already exists, update it with:

```sh
conda env update -f environment.yaml --prune
conda activate benchmarkOR
```

1. Solver-based evaluation requires a working Gurobi installation and a valid license in addition to the packages listed in [environment.yaml](environment.yaml). If you are not already inside the `benchmarkOR` environment, the benchmark scripts can be pointed to it explicitly with `--conda_env benchmarkOR`.

## Repository Layout

The main directories relevant to benchmarking are:

1. [data](data): raw benchmark data restored from the GitHub Release archive
2. [evaluation](evaluation): dataset-specific evaluation entry points
3. [scripts/test](scripts/test): recommended one-command test entry points
4. [scripts/run_eval.sh](scripts/run_eval.sh): lower-level orchestration script
5. [results](results): raw outputs and summary outputs
