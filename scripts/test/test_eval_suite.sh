#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

PROFILE=${PROFILE:-minimal}
MODELS=${MODELS:-"deepseek-v4-flash"}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-}
CONDA_ENV=${CONDA_ENV:-myenv}
SUMMARY_TAG=${SUMMARY_TAG:-}
SKIP_RUN=${SKIP_RUN:-false}
SKIP_SUMMARY=${SKIP_SUMMARY:-false}
SKIP_MISSING_RESULTS=${SKIP_MISSING_RESULTS:-false}
RERUN=${RERUN:-false}
VERBOSE=${VERBOSE:-false}
FAIL_FAST=${FAIL_FAST:-false}

usage() {
    cat <<'EOF'
Usage:
  sh scripts/test/test_eval_suite.sh [options]

Purpose:
  One-command entry for testing the full benchmark pipeline across all five targets.

Profiles:
  minimal   Run all five targets on a tiny slice: start=0, end=2, bench4opt_max_samples=2.
  full      Run all five targets without slice limits.

Options:
  --profile minimal|full      Default: minimal
  --models MODEL [MODEL ...]  Default: deepseek-v4-flash
  --conda_env ENV             Default: myenv
  --openai_api_key KEY        Optional; forwarded to scripts/run_eval_suite.sh
  --openai_base_url URL       Optional; forwarded to scripts/run_eval_suite.sh
  --summary_tag TAG           Optional summary directory name
  --skip_run                  Only summarize existing results
  --skip_summary              Run evaluation only, skip summary
  --skip_missing_results      Ignore missing result files when summarizing
  --rerun                     Force rerun for supported targets
  --verbose                   Forward verbose mode
  --fail_fast                 Stop immediately on the first failed command
  -h, --help                  Show this help

Examples:
  sh scripts/test/test_eval_suite.sh --profile minimal
  OPENAI_API_KEY=... OPENAI_BASE_URL=... sh scripts/test/test_eval_suite.sh --profile full --models deepseek-v3-0324 deepseek-v4-flash
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

while [ $# -gt 0 ]; do
    case "$1" in
        --profile)
            PROFILE=$2
            shift 2
            ;;
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
        --conda_env)
            CONDA_ENV=$2
            shift 2
            ;;
        --openai_api_key)
            OPENAI_API_KEY=$2
            shift 2
            ;;
        --openai_base_url)
            OPENAI_BASE_URL=$2
            shift 2
            ;;
        --summary_tag)
            SUMMARY_TAG=$2
            shift 2
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
        --rerun)
            RERUN=true
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

START=0
END=
BENCH4OPT_MAX_SAMPLES=

case "$PROFILE" in
    minimal)
        START=0
        END=2
        BENCH4OPT_MAX_SAMPLES=2
        if [ -z "$SUMMARY_TAG" ]; then
            SUMMARY_TAG="eval_suite_minimal"
        fi
        ;;
    full)
        START=0
        END=
        BENCH4OPT_MAX_SAMPLES=
        if [ -z "$SUMMARY_TAG" ]; then
            SUMMARY_TAG="eval_suite_full"
        fi
        ;;
    *)
        echo "Unsupported profile: $PROFILE" >&2
        exit 1
        ;;
esac

set -- sh scripts/run_eval_suite.sh \
    --models

for model_name in $MODELS; do
    set -- "$@" "$model_name"
done

set -- "$@" \
    --targets nl4opt_solver optibench_solver miplib_solver miplib_orgeval bench4opt_orgeval \
    --start "$START" \
    --summary_tag "$SUMMARY_TAG"

if [ -n "$END" ]; then
    set -- "$@" --end "$END"
fi

if [ -n "$BENCH4OPT_MAX_SAMPLES" ]; then
    set -- "$@" --bench4opt_max_samples "$BENCH4OPT_MAX_SAMPLES"
fi

if [ -n "$CONDA_ENV" ]; then
    set -- "$@" --conda_env "$CONDA_ENV"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    set -- "$@" --openai_api_key "$OPENAI_API_KEY"
fi

if [ -n "$OPENAI_BASE_URL" ]; then
    set -- "$@" --openai_base_url "$OPENAI_BASE_URL"
fi

if is_true "$SKIP_RUN"; then
    set -- "$@" --skip_run
fi

if is_true "$SKIP_SUMMARY"; then
    set -- "$@" --skip_summary
fi

if is_true "$SKIP_MISSING_RESULTS"; then
    set -- "$@" --skip_missing_results
fi

if is_true "$RERUN"; then
    set -- "$@" --rerun
fi

if is_true "$VERBOSE"; then
    set -- "$@" --verbose
fi

if is_true "$FAIL_FAST"; then
    set -- "$@" --fail_fast
fi

echo "Running benchmark test profile: $PROFILE"
echo "$*"
exec "$@"