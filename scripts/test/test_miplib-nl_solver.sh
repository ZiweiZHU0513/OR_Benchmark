#! /bin/bash

set -x

dataset_root="./data/miplib-nl"
openai_base_url="https://api.uniapi.io/v1"
openai_api_key="${OPENAI_API_KEY:-}"

if [ -z "$openai_api_key" ]; then
    echo "OPENAI_API_KEY is required" >&2
    exit 1
fi

models=(
    "deepseek-v3-0324"
    "deepseek-v4-flash"
    "deepseek-v4-pro"
)

temperature=0.6
top_p=1.0
max_tokens=32000
seed=42
verbose=true
start=0
end=none
timeout=600
tolerance=1e-4
rel_tolerance=1e-3
num_workers=16

for model_name in "${models[@]}"; do
    base_model_name=$(basename $model_name)

    if [ $start != 0 ] || [ $end != none ]; then
        save_path="./results/miplib-nl/${base_model_name}_${start}_${end}_solver.json"
    else
        save_path="./results/miplib-nl/${base_model_name}_solver.json"
    fi
    mkdir -p $(dirname $save_path)

    python -m evaluation.miplib-nl.run_evaluation_solver \
        --dataset_root $dataset_root \
        --model_name $model_name \
        --num_workers $num_workers \
        --temperature $temperature \
        --top_p $top_p \
        --max_tokens $max_tokens \
        --seed $seed \
        --openai_api_key $openai_api_key \
        --openai_base_url $openai_base_url \
        --start $start \
        $([ "$end" != "none" ] && echo "--end $end") \
        --timeout $timeout \
        --tolerance $tolerance \
        --rel_tolerance $rel_tolerance \
        --save_path $save_path \
        --verbose $verbose
done
