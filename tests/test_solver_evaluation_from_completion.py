import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from tqdm import tqdm

# Ensure project root is importable when running as module/script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate._lp_utils import extract_python_code
import evaluate.evaluate_code as evaluate_code


def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def evaluate_one(sample: Dict, data_dir: str, model_name: str, verbose: bool) -> Dict:
    completion = sample.get("completion", "")
    code = extract_python_code(completion)

    sample_id = str(sample.get("id", ""))
    problem_id = None
    if sample_id.startswith("BENCH4OPT_"):
        try:
            problem_id = sample_id.split("_")[1]
        except Exception:
            problem_id = None

    data_path = os.path.join(data_dir, sample["data_path"])
    reference_lp_path = os.path.join(data_dir, sample["reference_lp_path"])

    code_eval_result, solver_eval_result, solver_check_time = evaluate_code.evaluate_code_solver(
        model_name=model_name,
        data_dir=data_dir,
        code=code,
        data_path=data_path,
        reference_lp_path=reference_lp_path,
        verbose=verbose,
        problem_id=problem_id,
    )

    if code_eval_result["success"]:
        code_success = True
        code_error_msg = ""
        lp_path = code_eval_result["lp_file_path"]
        if solver_eval_result["success"]:
            solver_success = True
            solver_error_msg = solver_eval_result["message"]
        else:
            solver_success = False
            solver_error_msg = solver_eval_result["message"]
    else:
        code_success = False
        code_error_msg = code_eval_result["message"]
        lp_path = code_eval_result["lp_file_path"]
        solver_success = False
        solver_error_msg = ""

    sample["code"] = lp_path
    sample["reward"] = {
        "code_reward": 1.0 if code_success else 0.0,
        "solver_reward": 1.0 if solver_success else 0.0,
    }
    sample["verification"] = {
        "code_verification": code_error_msg,
        "solver_verification": solver_error_msg,
    }
    sample["solver_check_time"] = solver_check_time
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cached completions with solver only (no regeneration)."
    )
    parser.add_argument("--model_name", type=str, default="deepseek-v4-flash")
    parser.add_argument("--resume_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/bench4opt_mix_final")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    samples = load_results(args.resume_file)
    print(f"Loaded {len(samples)} samples from {args.resume_file}")
    print(f"Using data dir: {data_dir}")

    all_results: List[Dict] = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Solver evaluating"):
        batch = samples[i : i + args.batch_size]
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(
                executor.map(
                    lambda s: evaluate_one(s, data_dir, args.model_name, args.verbose),
                    batch,
                )
            )
        all_results.extend(results)

        # Incremental save to avoid losing progress.
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Solver evaluation results saved to {args.save_path}")


if __name__ == "__main__":
    main()
