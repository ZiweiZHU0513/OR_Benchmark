#!/bin/bash

set -euo pipefail
set -x

data_dir="${DATA_DIR:-data/bench4opt}"
model_name="${MODEL_NAME:-deepseek-v3-0324}"
openai_api_key="${OPENAI_API_KEY:-your key}"
openai_base_url="${OPENAI_BASE_URL:-}"
output_filename="${OUTPUT_FILENAME:-}"
max_workers="${MAX_WORKERS:-32}"
eval_workers="${EVAL_WORKERS:-1}"
eval_timeout="${EVAL_TIMEOUT:-180}"
save_every="${SAVE_EVERY:-32}"
max_samples="${MAX_SAMPLES:-}"
seed="${SEED:-42}"
temperature="${TEMPERATURE:-}"
top_p="${TOP_P:-}"
max_tokens="${MAX_TOKENS:-}"

while getopts "d:o:w:n:m:e:t:" opt; do
    case $opt in
        d) data_dir="$OPTARG" ;;
        o) output_filename="$OPTARG" ;;
        w) max_workers="$OPTARG" ;;
        n) save_every="$OPTARG" ;;
        m) max_samples="$OPTARG" ;;
        e) eval_workers="$OPTARG" ;;
        t) eval_timeout="$OPTARG" ;;
    esac
done

if [ -z "$output_filename" ]; then
    fixed_filename="test_pipeline_model_${model_name//\//-}_${data_dir//\//-}_resume_fast.json"
    log_filename="test_pipeline_model_${model_name//\//-}_${data_dir//\//-}_resume_fast.log"
else
    fixed_filename="$output_filename"
    log_filename="${output_filename%.json}.log"
fi

mkdir -p outputs

cmd=(
    python -m evaluation.bench4opt.run_evaluation_fast
    --model_name "$model_name"
    --save_path "outputs/$fixed_filename"
    --data_dir "$data_dir"
    --openai_api_key "$openai_api_key"
    --openai_base_url "$openai_base_url"
    --max_workers "$max_workers"
    --eval_workers "$eval_workers"
    --eval_timeout "$eval_timeout"
    --save_every "$save_every"
    --seed "$seed"
)

if [ -n "$max_samples" ]; then
    cmd+=(--max_samples "$max_samples")
fi

if [ -n "$temperature" ]; then
    cmd+=(--temperature "$temperature")
fi

if [ -n "$top_p" ]; then
    cmd+=(--top_p "$top_p")
fi

if [ -n "$max_tokens" ]; then
    cmd+=(--max_tokens "$max_tokens")
fi

"${cmd[@]}" 2>&1 | tee "outputs/$log_filename"