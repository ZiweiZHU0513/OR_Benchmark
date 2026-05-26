#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

MODELS=${MODELS:-"deepseek-v3-0324 deepseek-v4-flash"}
TARGETS=${TARGETS:-"nl4opt_solver optibench_solver miplib_solver miplib_orgeval bench4opt_orgeval"}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.uniapi.io/v1}
CONDA_ENV=${CONDA_ENV:-}
RESULT_ROOT=${RESULT_ROOT:-results}
SUMMARY_ROOT=${SUMMARY_ROOT:-}
SUMMARY_TAG=${SUMMARY_TAG:-}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-1.0}
MAX_TOKENS=${MAX_TOKENS:-32000}
SEED=${SEED:-42}
START=${START:-0}
END=${END:-}
BENCH4OPT_MAX_SAMPLES=${BENCH4OPT_MAX_SAMPLES:-}
NL4OPT_DATASET=${NL4OPT_DATASET:-./data/NL4OPT}
OPTIBENCH_DATASET=${OPTIBENCH_DATASET:-./data/optibench}
MIPLIB_SOLVER_DATASET=${MIPLIB_SOLVER_DATASET:-data/miplib-nl}
MIPLIB_ORGEVAL_DATASET=${MIPLIB_ORGEVAL_DATASET:-data/miplib-nl_exclude_failure}
MIPLIB_SOLVER_SAVE_EVERY=${MIPLIB_SOLVER_SAVE_EVERY:-8}
BENCH4OPT_DATASET=${BENCH4OPT_DATASET:-data/bench4opt}
RERUN=${RERUN:-false}
SKIP_RUN=${SKIP_RUN:-false}
SKIP_SUMMARY=${SKIP_SUMMARY:-false}
SKIP_MISSING_RESULTS=${SKIP_MISSING_RESULTS:-false}
VERBOSE=${VERBOSE:-false}
FAIL_FAST=${FAIL_FAST:-false}

FAILED_COMMANDS=0

usage() {
    cat <<'EOF'
Usage:
  sh scripts/run_eval_suite.sh [options]

Recommended entry for daily use:
    sh scripts/test/test_eval_suite.sh --profile minimal
    sh scripts/test/test_eval_suite.sh --profile full

Options:
  --models MODEL [MODEL ...]          Models to evaluate
  --targets TARGET [TARGET ...]       Targets to run: nl4opt_solver optibench_solver miplib_solver miplib_orgeval bench4opt_orgeval
  --openai_api_key KEY                API key
  --openai_base_url URL               Base URL
  --conda_env ENV                     Optional conda env used for all python commands
  --result_root DIR                   Raw result root, default: results
  --summary_root DIR                  Summary output root
  --summary_tag TAG                   Summary subdirectory name
  --temperature VALUE                Shared temperature, default: 0.6
  --top_p VALUE                       Shared top_p, default: 1.0
  --max_tokens VALUE                  Shared max_tokens, default: 32000
  --seed VALUE                        Shared seed, default: 42
  --start VALUE                       Shared start index, default: 0
  --end VALUE                         Shared end index for nl4opt/optibench/miplib; pass none to disable
  --bench4opt_max_samples VALUE       Optional max_samples for bench4opt orgeval
  --nl4opt_dataset PATH               Default: ./data/NL4OPT
  --optibench_dataset PATH            Default: ./data/optibench
  --miplib_solver_dataset PATH        Default: data/miplib-nl
  --miplib_orgeval_dataset PATH       Default: data/miplib-nl_exclude_failure
  --bench4opt_dataset PATH            Default: data/bench4opt
  --rerun                             Force rerun for solver and miplib orgeval targets
  --skip_run                          Do not launch evaluations, only summarize existing results
  --skip_summary                      Do not generate summary after evaluation
  --skip_missing_results              Skip missing raw result files during summary generation
  --verbose                           Forward verbose=true to dataset evaluators
  --fail_fast                         Stop immediately when one command fails
  -h, --help                          Show this help

Environment variable usage is also supported, e.g.:
  OPENAI_API_KEY=... OPENAI_BASE_URL=... MODELS="deepseek-v3-0324 deepseek-v4-flash" sh scripts/run_eval_suite.sh
EOF
}

is_true() {
    value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$value" in
        1|true|yes|y|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

safe_model_name() {
    printf '%s' "$1" | sed 's#[^A-Za-z0-9._-]#-#g'
}

range_suffix() {
    if [ "$START" != "0" ] || [ -n "$END" ]; then
        end_text=${END:-none}
        printf '_%s_%s' "$START" "$end_text"
    fi
}

result_path_for_target() {
    target_key=$1
    model_name=$2
    safe_model=$(safe_model_name "$model_name")
    slice_suffix=$(range_suffix)

    case "$target_key" in
        nl4opt_solver)
            printf '%s/nl4opt/%s%s_solver.json' "$RESULT_ROOT" "$safe_model" "$slice_suffix"
            ;;
        optibench_solver)
            printf '%s/optibench/%s%s_solver.json' "$RESULT_ROOT" "$safe_model" "$slice_suffix"
            ;;
        miplib_solver)
            printf '%s/miplib-nl/%s%s_solver.json' "$RESULT_ROOT" "$safe_model" "$slice_suffix"
            ;;
        miplib_orgeval)
            printf '%s/miplib-nl/%s%s_orgeval.json' "$RESULT_ROOT" "$safe_model" "$slice_suffix"
            ;;
        bench4opt_orgeval)
            max_suffix=
            if [ -n "$BENCH4OPT_MAX_SAMPLES" ]; then
                max_suffix="_max${BENCH4OPT_MAX_SAMPLES}"
            fi
            printf '%s/bench4opt/%s%s_orgeval.json' "$RESULT_ROOT" "$safe_model" "$max_suffix"
            ;;
        *)
            printf '%s' ""
            ;;
    esac
}

ensure_parent_dir() {
    parent_dir=$(dirname "$1")
    mkdir -p "$parent_dir"
}

run_python() {
    if [ -n "$CONDA_ENV" ]; then
        conda run -n "$CONDA_ENV" python "$@"
    else
        python "$@"
    fi
}

run_command() {
    label=$1
    shift

    echo
    echo "=== Running ${label} ==="
    echo "python $*"

    if run_python "$@"; then
        return 0
    else
        rc=$?
    fi

    FAILED_COMMANDS=$((FAILED_COMMANDS + 1))
    echo "[WARN] ${label} failed with exit code ${rc}" >&2
    if is_true "$FAIL_FAST"; then
        exit "$rc"
    fi
    return 0
}

run_nl4opt_solver() {
    model_name=$1
    save_path=$2

    set -- -m evaluation.nl4opt.run_evaluation_solver \
        --dataset_name "$NL4OPT_DATASET" \
        --split test \
        --batch_size 32 \
        --num_workers 32 \
        --model_name "$model_name" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --seed "$SEED" \
        --openai_api_key "$OPENAI_API_KEY" \
        --openai_base_url "$OPENAI_BASE_URL" \
        --start "$START" \
        --timeout 360.0 \
        --tolerance 1e-6 \
        --save_path "$save_path" \
        --verbose "$VERBOSE"

    if [ -n "$END" ]; then
        set -- "$@" --end "$END"
    fi
    if is_true "$RERUN"; then
        set -- "$@" --rerun
    fi

    run_command "nl4opt solver / ${model_name}" "$@"
}

run_optibench_solver() {
    model_name=$1
    save_path=$2

    set -- -m evaluation.optibench.run_evaluation_solver \
        --dataset_name "$OPTIBENCH_DATASET" \
        --split test \
        --batch_size 32 \
        --num_workers 32 \
        --model_name "$model_name" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --seed "$SEED" \
        --openai_api_key "$OPENAI_API_KEY" \
        --openai_base_url "$OPENAI_BASE_URL" \
        --start "$START" \
        --timeout 360.0 \
        --tolerance 1e-6 \
        --save_path "$save_path" \
        --verbose "$VERBOSE"

    if [ -n "$END" ]; then
        set -- "$@" --end "$END"
    fi
    if is_true "$RERUN"; then
        set -- "$@" --rerun
    fi

    run_command "optibench solver / ${model_name}" "$@"
}

run_miplib_solver() {
    model_name=$1
    save_path=$2

    set -- -m evaluation.miplib-nl.run_evaluation_solver \
        --dataset_root "$MIPLIB_SOLVER_DATASET" \
        --model_name "$model_name" \
        --num_workers 16 \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --seed "$SEED" \
        --openai_api_key "$OPENAI_API_KEY" \
        --openai_base_url "$OPENAI_BASE_URL" \
        --start "$START" \
        --timeout 600 \
        --tolerance 1e-6 \
        --rel_tolerance 1e-6 \
        --save_every "$MIPLIB_SOLVER_SAVE_EVERY" \
        --save_path "$save_path" \
        --verbose "$VERBOSE"

    if [ -n "$END" ]; then
        set -- "$@" --end "$END"
    fi
    if is_true "$RERUN"; then
        set -- "$@" --rerun
    fi

    run_command "miplib-nl solver / ${model_name}" "$@"
}

run_miplib_orgeval() {
    model_name=$1
    save_path=$2

    set -- -m evaluation.miplib-nl.run_evaluation_fast \
        --dataset_root "$MIPLIB_ORGEVAL_DATASET" \
        --model_name "$model_name" \
        --num_workers 8 \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --seed "$SEED" \
        --openai_api_key "$OPENAI_API_KEY" \
        --openai_base_url "$OPENAI_BASE_URL" \
        --start "$START" \
        --timeout 180 \
        --save_path "$save_path" \
        --verbose "$VERBOSE"

    if [ -n "$END" ]; then
        set -- "$@" --end "$END"
    fi
    if is_true "$RERUN"; then
        set -- "$@" --rerun
    fi

    run_command "miplib-nl orgeval / ${model_name}" "$@"
}

run_bench4opt_orgeval() {
    model_name=$1
    save_path=$2

    set -- -m evaluation.bench4opt.run_evaluation_fast \
        --model_name "$model_name" \
        --save_path "$save_path" \
        --data_dir "$BENCH4OPT_DATASET" \
        --openai_api_key "$OPENAI_API_KEY" \
        --openai_base_url "$OPENAI_BASE_URL" \
        --max_workers 32 \
        --eval_workers 1 \
        --eval_timeout 180 \
        --save_every 32 \
        --seed "$SEED" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS"

    if [ -n "$BENCH4OPT_MAX_SAMPLES" ]; then
        set -- "$@" --max_samples "$BENCH4OPT_MAX_SAMPLES"
    fi

    run_command "bench4opt orgeval / ${model_name}" "$@"
}

run_summary() {
    set -- "$SCRIPT_DIR/run_eval_suite.py" --skip_run --result_root "$RESULT_ROOT"

    if [ -n "$SUMMARY_ROOT" ]; then
        set -- "$@" --summary_root "$SUMMARY_ROOT"
    fi
    if [ -n "$SUMMARY_TAG" ]; then
        set -- "$@" --summary_tag "$SUMMARY_TAG"
    fi
    if [ "$START" != "0" ]; then
        set -- "$@" --start "$START"
    fi
    if [ -n "$END" ]; then
        set -- "$@" --end "$END"
    fi
    if [ -n "$BENCH4OPT_MAX_SAMPLES" ]; then
        set -- "$@" --bench4opt_max_samples "$BENCH4OPT_MAX_SAMPLES"
    fi

    set -- "$@" \
        --nl4opt_dataset "$NL4OPT_DATASET" \
        --optibench_dataset "$OPTIBENCH_DATASET" \
        --miplib_solver_dataset "$MIPLIB_SOLVER_DATASET" \
        --miplib_orgeval_dataset "$MIPLIB_ORGEVAL_DATASET" \
        --bench4opt_dataset "$BENCH4OPT_DATASET"

    if is_true "$SKIP_MISSING_RESULTS"; then
        set -- "$@" --skip_missing_results
    fi

    set -- "$@" --models
    for model_name in $MODELS; do
        set -- "$@" "$model_name"
    done

    set -- "$@" --targets
    for target_key in $TARGETS; do
        set -- "$@" "$target_key"
    done

    run_command "summary" "$@"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --models)
            shift
            MODELS=
            while [ $# -gt 0 ]; do
                case "$1" in
                    --*)
                        break
                        ;;
                    *)
                        if [ -n "$MODELS" ]; then
                            MODELS="$MODELS $1"
                        else
                            MODELS=$1
                        fi
                        shift
                        ;;
                esac
            done
            continue
            ;;
        --targets)
            shift
            TARGETS=
            while [ $# -gt 0 ]; do
                case "$1" in
                    --*)
                        break
                        ;;
                    *)
                        if [ -n "$TARGETS" ]; then
                            TARGETS="$TARGETS $1"
                        else
                            TARGETS=$1
                        fi
                        shift
                        ;;
                esac
            done
            continue
            ;;
        --openai_api_key)
            OPENAI_API_KEY=$2
            shift 2
            ;;
        --openai_base_url)
            OPENAI_BASE_URL=$2
            shift 2
            ;;
        --conda_env)
            CONDA_ENV=$2
            shift 2
            ;;
        --result_root)
            RESULT_ROOT=$2
            shift 2
            ;;
        --summary_root)
            SUMMARY_ROOT=$2
            shift 2
            ;;
        --summary_tag)
            SUMMARY_TAG=$2
            shift 2
            ;;
        --temperature)
            TEMPERATURE=$2
            shift 2
            ;;
        --top_p)
            TOP_P=$2
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS=$2
            shift 2
            ;;
        --seed)
            SEED=$2
            shift 2
            ;;
        --start)
            START=$2
            shift 2
            ;;
        --end)
            if [ "$2" = "none" ]; then
                END=
            else
                END=$2
            fi
            shift 2
            ;;
        --bench4opt_max_samples)
            BENCH4OPT_MAX_SAMPLES=$2
            shift 2
            ;;
        --nl4opt_dataset)
            NL4OPT_DATASET=$2
            shift 2
            ;;
        --optibench_dataset)
            OPTIBENCH_DATASET=$2
            shift 2
            ;;
        --miplib_solver_dataset)
            MIPLIB_SOLVER_DATASET=$2
            shift 2
            ;;
        --miplib_orgeval_dataset)
            MIPLIB_ORGEVAL_DATASET=$2
            shift 2
            ;;
        --bench4opt_dataset)
            BENCH4OPT_DATASET=$2
            shift 2
            ;;
        --rerun)
            RERUN=true
            shift
            ;;
        --skip_run)
            SKIP_RUN=true
            shift
            ;;
        --skip_summary)
            SKIP_SUMMARY=true
            shift
            ;;
        --skip_missing_results)
            SKIP_MISSING_RESULTS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --fail_fast)
            FAIL_FAST=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$MODELS" ]; then
    echo "MODELS is empty" >&2
    exit 1
fi

if ! is_true "$SKIP_RUN"; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "OPENAI_API_KEY is required unless --skip_run is used" >&2
        exit 1
    fi
    if [ -z "$OPENAI_BASE_URL" ]; then
        echo "OPENAI_BASE_URL is required unless --skip_run is used" >&2
        exit 1
    fi
fi

if ! is_true "$SKIP_RUN"; then
    for target_key in $TARGETS; do
        for model_name in $MODELS; do
            save_path=$(result_path_for_target "$target_key" "$model_name")
            ensure_parent_dir "$save_path"

            case "$target_key" in
                nl4opt_solver)
                    run_nl4opt_solver "$model_name" "$save_path"
                    ;;
                optibench_solver)
                    run_optibench_solver "$model_name" "$save_path"
                    ;;
                miplib_solver)
                    run_miplib_solver "$model_name" "$save_path"
                    ;;
                miplib_orgeval)
                    run_miplib_orgeval "$model_name" "$save_path"
                    ;;
                bench4opt_orgeval)
                    run_bench4opt_orgeval "$model_name" "$save_path"
                    ;;
                *)
                    echo "Unknown target: $target_key" >&2
                    exit 1
                    ;;
            esac
        done
    done
fi

if ! is_true "$SKIP_SUMMARY"; then
    run_summary
fi

if [ "$FAILED_COMMANDS" -gt 0 ]; then
    echo "Completed with ${FAILED_COMMANDS} failed command(s)" >&2
    exit 1
fi
