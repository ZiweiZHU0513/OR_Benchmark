#! /bin/bash

set -x

dataset_name="./data/NL4OPT"
split="test"

use_vllm=false
if [ $use_vllm = false ]; then
    model_name="deepseek-v4-pro"
    openai_base_url=""
    openai_api_key="${OPENAI_API_KEY:-}"
    batch_size=32
else
    model_name="qwen3-4b-short-cot_v3"
    openai_base_url='http://localhost:6001/v1'
    openai_api_key=none
    batch_size=32
fi

if [ "$use_vllm" = false ] && [ -z "$openai_api_key" ]; then
    echo "OPENAI_API_KEY is required when use_vllm=false" >&2
    exit 1
fi

temperature=0.6
top_p=1.0
max_tokens=32000
seed=42
verbose=true
start=0
end=none
timeout=360.0
tolerance=1e-6
num_workers=32
base_model_name=$(basename $model_name)

if [ $start != 0 ] || [ $end != none ]; then
    save_path="./results/nl4opt/${base_model_name}_${start}_${end}_solver.json"
else
    save_path="./results/nl4opt/${base_model_name}_solver.json"
fi
mkdir -p $(dirname $save_path)

python -m evaluation.nl4opt.run_evaluation_solver --rerun \
    --dataset_name $dataset_name \
    --split $split \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --model_name $model_name \
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
    --save_path $save_path \
    --verbose $verbose

