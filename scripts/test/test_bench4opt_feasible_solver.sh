#! /bin/bash

set -euo pipefail
set -x

data_dir="${DATA_DIR:-./data/bench4opt_feasible}"
model_name="${MODEL_NAME:-deepseek-v4-flash}"
openai_base_url="${OPENAI_BASE_URL:-}"
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
timeout="${TIMEOUT:-360}"
tolerance="${TOLERANCE:-1e-6}"
max_workers="${MAX_WORKERS:-32}"
save_every="${SAVE_EVERY:-32}"
base_model_name=$(basename "$model_name")

if [ "$start" != "0" ] || [ "$end" != "none" ]; then
    save_path="./results/bench4opt/${base_model_name}_${start}_${end}_solver.json"
else
    save_path="./results/bench4opt/${base_model_name}_solver.json"
fi

mkdir -p "$(dirname "$save_path")"

python -m evaluation.bench4opt.run_evaluation_solver --rerun \
    --model_name "$model_name" \
    --save_path "$save_path" \
    --data_dir "$data_dir" \
    --openai_api_key "$openai_api_key" \
    --openai_base_url "$openai_base_url" \
    --temperature "$temperature" \
    --top_p "$top_p" \
    --max_tokens "$max_tokens" \
    --max_workers "$max_workers" \
    --timeout "$timeout" \
    --tolerance "$tolerance" \
    --save_every "$save_every" \
    --seed "$seed" \
    --start "$start" \
    $([ "$end" != "none" ] && echo "--end $end") \
    --verbose "$verbose"