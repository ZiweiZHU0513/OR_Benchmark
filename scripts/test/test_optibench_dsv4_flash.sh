#! /bin/bash

set -euo pipefail
set -x

dataset_name="./data/optibench"
split="test"

model_name="deepseek-v4-flash"
openai_base_url="https://api.uniapi.io/v1"
openai_api_key="${OPENAI_API_KEY:-}"

if [ -z "$openai_api_key" ]; then
    echo "OPENAI_API_KEY is required" >&2
    exit 1
fi

batch_size="${BATCH_SIZE:-32}"
temperature="${TEMPERATURE:-0.6}"
top_p="${TOP_P:-1.0}"
max_tokens="${MAX_TOKENS:-32000}"
seed="${SEED:-42}"
verbose="${VERBOSE:-true}"
start="${START:-0}"
end="${END:-none}"
timeout="${TIMEOUT:-360.0}"
tolerance="${TOLERANCE:-1e-6}"
num_workers="${NUM_WORKERS:-32}"
base_model_name=$(basename "$model_name")

if [ "$start" != "0" ] || [ "$end" != "none" ]; then
    save_path="./results/optibench/${base_model_name}_${start}_${end}_solver.json"
else
    save_path="./results/optibench/${base_model_name}_solver.json"
fi

mkdir -p "$(dirname "$save_path")"

python -m evaluation.optibench.run_evaluation_solver --rerun \
    --dataset_name "$dataset_name" \
    --split "$split" \
    --batch_size "$batch_size" \
    --num_workers "$num_workers" \
    --model_name "$model_name" \
    --temperature "$temperature" \
    --top_p "$top_p" \
    --max_tokens "$max_tokens" \
    --seed "$seed" \
    --openai_api_key "$openai_api_key" \
    --openai_base_url "$openai_base_url" \
    --start "$start" \
    $([ "$end" != "none" ] && echo "--end $end") \
    --timeout "$timeout" \
    --tolerance "$tolerance" \
    --save_path "$save_path" \
    --verbose "$verbose"