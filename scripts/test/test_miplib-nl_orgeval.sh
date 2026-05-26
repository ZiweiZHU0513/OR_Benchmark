#! /bin/bash

set -euo pipefail
set -x

dataset_root="${DATASET_ROOT:-./data/miplib-nl_exclude_failure}"
model_name="${MODEL_NAME:-deepseek-v4-flash}"
openai_base_url="https://api.uniapi.io/v1"
openai_api_key="${OPENAI_API_KEY:-}"

if [ -z "$openai_api_key" ]; then
    echo "OPENAI_API_KEY is required" >&2
    exit 1
fi

temperature="${TEMPERATURE:-0.6}"
top_p="${TOP_P:-1.0}"
max_tokens="${MAX_TOKENS:-32000}"
seed="${SEED:-42}"
verbose="${VERBOSE:-true}"
start="${START:-0}"
end="${END:-none}"
timeout="${TIMEOUT:-180}"
num_workers="${NUM_WORKERS:-8}"

base_model_name="${model_name//\//-}"

if [ "$start" != "0" ] || [ "$end" != "none" ]; then
    save_path="./results/miplib-nl/${base_model_name}_${start}_${end}_orgeval.json"
else
    save_path="./results/miplib-nl/${base_model_name}_orgeval.json"
fi

mkdir -p "$(dirname "$save_path")"

python -m evaluation.miplib-nl.run_evaluation_fast \
    --dataset_root "$dataset_root" \
    --model_name "$model_name" \
    --num_workers "$num_workers" \
    --temperature "$temperature" \
    --top_p "$top_p" \
    --max_tokens "$max_tokens" \
    --seed "$seed" \
    --openai_api_key "$openai_api_key" \
    --openai_base_url "$openai_base_url" \
    --start "$start" \
    $([ "$end" != "none" ] && echo "--end $end") \
    --timeout "$timeout" \
    --save_path "$save_path" \
    --verbose "$verbose"