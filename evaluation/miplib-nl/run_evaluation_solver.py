"""
miplib-nl solver-based evaluation.

For each instance directory under data/miplib-nl/, render the prompt template
with fields from instance.json (abstract_problem, parameters, files) and the
absolute path of the instance directory. Then call the LLM to generate gurobi
code, execute it with cwd=instance_dir (so ./data/*.csv paths resolve), and
compare ObjVal vs the GT optimal_value.

Supports resume (drops api_error entries on reload).
"""
import os
import sys
import json
import time
import argparse
import threading
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from tqdm import tqdm
from openai import OpenAI

# Package directory contains a hyphen, so use importlib instead of a static import.
_solver_eval_mod = importlib.import_module("evaluation.miplib-nl.solver_evaluation")
evaluate_instance = _solver_eval_mod.evaluate_instance


PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompt_template.txt"
)


def load_prompt_template() -> str:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(template: str, instance: dict, instance_dir: str) -> str:
    problem_desc = instance.get("abstract_problem", "")
    params = instance.get("parameters", {})
    files = instance.get("files", None)

    params_json = json.dumps(params, ensure_ascii=False, indent=2)
    if files:
        files_info = json.dumps(files, ensure_ascii=False, indent=2)
    else:
        files_info = "No external data files for this problem. All parameters are provided inline above."

    prompt = template
    prompt = prompt.replace("[Problem Description]", problem_desc)
    prompt = prompt.replace("[Parameters JSON]", params_json)
    prompt = prompt.replace("[Data Files Info]", files_info)
    prompt = prompt.replace("[Working Directory]", instance_dir)
    return prompt


def list_instances(dataset_root: str) -> list:
    """Return list of (instance_id, instance_dir, instance_json) sorted by id."""
    items = []
    for name in sorted(os.listdir(dataset_root)):
        inst_dir = os.path.join(dataset_root, name)
        inst_json = os.path.join(inst_dir, "instance.json")
        if not os.path.isdir(inst_dir):
            continue
        if not os.path.isfile(inst_json):
            continue
        items.append((name, inst_dir, inst_json))
    return items


def generate_completion(client, messages, model_name, **kwargs):
    from openai import InternalServerError, APIError
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=messages, **kwargs
            )
            return resp.choices[0].message.content
        except (InternalServerError, APIError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def evaluate_one(
    sample: dict,
    client,
    model_name: str,
    prompt_template: str,
    gen_kwargs: dict,
    timeout: int,
    tolerance: float,
    rel_tolerance: float,
    verbose: bool,
) -> dict:
    sample_id = sample["sample_id"]
    instance_dir = sample["instance_dir"]

    # 1. Load instance (data error -> not retried)
    try:
        with open(sample["instance_json"]) as f:
            instance = json.load(f)
    except Exception as e:
        return {
            "sample_id": sample_id,
            "success": False,
            "predicted_optimal_value": None,
            "gt_optimal_value": None,
            "msg": f"DATA_ERROR: failed to load instance.json: {e}",
            "completion": "",
            "prompt_chars": 0,
            "eval_time": 0.0,
            "api_error": False,
        }

    gt_raw = instance.get("optimal_value")
    if gt_raw is None:
        return {
            "sample_id": sample_id,
            "success": False,
            "predicted_optimal_value": None,
            "gt_optimal_value": None,
            "msg": "DATA_ERROR: No optimal_value in instance.json",
            "completion": "",
            "prompt_chars": 0,
            "eval_time": 0.0,
            "api_error": False,
        }
    # Numeric GT if convertible, otherwise keep as categorical string
    if isinstance(gt_raw, (int, float)):
        gt = float(gt_raw)
    else:
        try:
            gt = float(gt_raw)
        except (TypeError, ValueError):
            gt = gt_raw  # e.g. "infeasible", "unbounded", "impossible"

    # 2. Render prompt + call LLM (only LLM errors -> api_error=True)
    try:
        prompt = render_prompt(prompt_template, instance, instance_dir)
        messages = [{"role": "user", "content": prompt}]
        completion = generate_completion(client, messages, model_name, **gen_kwargs)
    except Exception as e:
        if verbose:
            import traceback; traceback.print_exc()
        return {
            "sample_id": sample_id,
            "success": False,
            "predicted_optimal_value": None,
            "gt_optimal_value": gt_raw,
            "msg": f"API_ERROR: {str(e)[:300]}",
            "completion": "",
            "prompt_chars": 0,
            "eval_time": 0.0,
            "api_error": True,
        }

    # 3. Execute & score (eval errors are deterministic -> not retried)
    try:
        eval_start = time.time()
        eval_result = evaluate_instance(
            code=completion,
            gt_optimal_value=gt,
            instance_dir=instance_dir,
            tolerance=tolerance,
            rel_tolerance=rel_tolerance,
            timeout=timeout,
            verbose=verbose,
        )
        eval_time = time.time() - eval_start
        return {
            "sample_id": sample_id,
            "success": eval_result["success"],
            "predicted_optimal_value": eval_result["predicted_optimal_value"],
            "predicted_status": eval_result.get("predicted_status"),
            "gt_optimal_value": eval_result["gt_optimal_value"],
            "error": eval_result.get("error"),
            "rel_error": eval_result.get("rel_error"),
            "msg": eval_result["message"],
            "completion": completion,
            "prompt_chars": len(prompt),
            "eval_time": eval_time,
            "api_error": False,
        }
    except Exception as e:
        if verbose:
            import traceback; traceback.print_exc()
        return {
            "sample_id": sample_id,
            "success": False,
            "predicted_optimal_value": None,
            "gt_optimal_value": gt_raw,
            "msg": f"EVAL_ERROR: {str(e)[:300]}",
            "completion": completion,
            "prompt_chars": len(prompt) if 'prompt' in locals() else 0,
            "eval_time": 0.0,
            "api_error": False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data/miplib-nl",
                        help="Path to miplib-nl benchmark root containing instance directories")
    parser.add_argument("--model_name", default="gpt-4o")
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--openai_base_url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-instance Gurobi solve timeout (seconds)")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Absolute tolerance for optimal value match")
    parser.add_argument("--rel_tolerance", type=float, default=1e-3,
                        help="Relative tolerance for optimal value match")
    parser.add_argument("--save_path", default="results/miplib-nl/gpt-4o_solver.json")
    parser.add_argument("--save_every", type=int, default=8,
                        help="Write intermediate results after every N completed samples; 0 disables incremental saves")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--verbose", type=lambda x: str(x).lower() == "true", default=False)
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=2))

    instances = list_instances(args.dataset_root)
    print(f"Discovered {len(instances)} instances under {args.dataset_root}")

    samples = [
        {"sample_id": name, "instance_dir": os.path.abspath(d), "instance_json": j}
        for (name, d, j) in instances
    ]
    end = args.end if args.end is not None else len(samples)
    samples = samples[args.start:end]
    print(f"Processing range: [{args.start}, {end})  -> {len(samples)} samples")

    output_data = []
    completed_ids = set()

    if args.save_path and os.path.exists(args.save_path) and not args.rerun:
        try:
            with open(args.save_path) as f:
                output_data = json.load(f)
            for item in output_data:
                if not item.get("api_error", False) and "success" in item:
                    completed_ids.add(item["sample_id"])
            output_data = [it for it in output_data if it.get("sample_id") in completed_ids]
            print(f"Resume mode: {len(completed_ids)} already done, api_error entries will be retried")
        except Exception as e:
            print(f"Warning: could not load existing results: {e}")
            output_data = []
            completed_ids = set()
    elif args.rerun:
        print("Rerun mode: processing all samples from scratch")
        output_data = []
        completed_ids = set()

    to_process = [s for s in samples if s["sample_id"] not in completed_ids]
    print(f"To process: {len(to_process)}")

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)
    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    prompt_template = load_prompt_template()

    results = []
    lock = threading.Lock()
    completed_since_last_save = 0

    def save_results_snapshot(current_results):
        if not args.save_path:
            return
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        tmp = args.save_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(output_data + current_results, f, indent=2)
        os.replace(tmp, args.save_path)

    def worker(sample):
        nonlocal completed_since_last_save
        r = evaluate_one(
            sample, client, args.model_name, prompt_template,
            gen_kwargs, args.timeout, args.tolerance, args.rel_tolerance, args.verbose,
        )
        with lock:
            results.append(r)
            completed_since_last_save += 1
            if args.save_path and args.save_every > 0 and completed_since_last_save >= args.save_every:
                save_results_snapshot(results)
                completed_since_last_save = 0
        return r

    if to_process:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(worker, s): s for s in to_process}
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Evaluating {args.model_name}",
                dynamic_ncols=True,
            )
            done = 0
            success = 0
            for fut in pbar:
                done += 1
                try:
                    res = fut.result()
                    if res and res.get("success"):
                        success += 1
                except Exception as e:
                    if args.verbose:
                        print(f"Worker error: {e}")
                pbar.set_postfix(acc=f"{success}/{done} ({success/done*100:.1f}%)")

    final = output_data + results
    if args.save_path:
        save_results_snapshot(results)
        print(f"Saved to {args.save_path}")

    total = len(final)
    n_success = sum(1 for r in final if r.get("success"))
    n_api_err = sum(1 for r in final if r.get("api_error"))
    fail_msgs = Counter(r.get("msg", "") for r in final if not r.get("success"))

    print(f"\n{'='*60}")
    print(f"Total: {total}  Success: {n_success}  Acc: {n_success/total*100:.2f}%  api_errors: {n_api_err}")
    print(f"{'='*60}")
    print("Top failure reasons:")
    for msg, c in fail_msgs.most_common(10):
        print(f"  [{c}] {msg[:160]}")


if __name__ == "__main__":
    main()
