import argparse
import json
import multiprocessing as mp
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set

from tqdm import tqdm

import evaluate.evaluate_code as evaluate_code
from evaluate._lp_utils import extract_python_code


PROMPT_PREFIX = """You are an expert in optimization modeling and gurobipy. Build a model for the following optimization problem and implement it using Gurobi Optimizer with the `gurobipy` package version 11.0.3. The requirements are as follows:

1. Do not optimize or solve the model. Instead, save the model as an `.lp` file.
2. Specify the upper and lower bounds for all variables.
3. Be careful with the data types and dimensions of all variables and parameters.
4. Do not use any extra parameters that are not provided in the data file. If additional parameters are necessary, define them clearly.
5. Output the Python code only. Do not output anything else.
6. Make sure your output code can be run directly in Gurobi Optimizer and saves the desired `.lp` file."""

_thread_local = threading.local()
_eval_semaphore = None


def _evaluate_completion_worker(data_dir: str, sample_data: dict, model_name: str, queue: mp.Queue):
    try:
        result = evaluate_completion(data_dir, sample_data, model_name)
        queue.put((True, result))
    except Exception as exc:
        queue.put((False, str(exc)))


def evaluate_completion_with_timeout(data_dir: str, sample_data: dict, model_name: str, timeout_seconds: int):
    if timeout_seconds <= 0:
        return evaluate_completion(data_dir, sample_data, model_name)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_evaluate_completion_worker,
        args=(data_dir, sample_data, model_name, queue),
    )
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"evaluation timeout after {timeout_seconds}s")

    if queue.empty():
        raise RuntimeError("evaluation process finished without returning result")

    success, payload = queue.get()
    if not success:
        raise RuntimeError(payload)
    return payload


def load_samples(data_dir: str) -> List[dict]:
    test_path = os.path.join(data_dir, "test.jsonl")
    with open(test_path, "r", encoding="utf-8") as handle:
        raw_samples = [json.loads(line) for line in handle]

    samples = []
    for sample in raw_samples:
        samples.append(
            {
                "id": sample["id"],
                "data_path": sample["data_path"],
                "problem": sample["problem"],
                "reference_lp_path": sample["reference_lp_path"],
                "problem_type": sample.get("problem_type"),
                "problem_class": sample.get("problem_class"),
            }
        )
    return samples


def generate_prompt(sample_data: dict) -> str:
    return f"{PROMPT_PREFIX}\n\n{sample_data['problem']}"


def get_openai_client():
    client = getattr(_thread_local, "openai_client", None)
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        _thread_local.openai_client = client
    return client


def generate_completion(sample_data: dict, model_name: str, **generation_kwargs) -> str:
    client = get_openai_client()

    kwargs = {
        "model": model_name,
        "seed": 42,
    }
    kwargs.update(generation_kwargs)

    request_output = client.chat.completions.create(
        messages=[{"role": "user", "content": sample_data["prompt"]}],
        **kwargs,
    )
    return request_output.choices[0].message.content


def evaluate_completion(data_dir: str, sample_data: dict, model_name: str):
    code = extract_python_code(sample_data["completion"])
    data_path = os.path.join(data_dir, sample_data["data_path"])
    reference_lp_path = os.path.join(data_dir, sample_data["reference_lp_path"])

    code_eval_result, wl_eval_result, equivalence_check_time = evaluate_code.evaluate_code_re(
        model_name=model_name,
        data_dir=data_dir,
        code=code,
        data_path=data_path,
        reference_lp_path=reference_lp_path,
        verbose=False,
        problem_id=sample_data["id"].split("_")[1],
    )

    if code_eval_result["success"]:
        code_success = True
        code_error_msg = ""
        code = code_eval_result["lp_file_path"]
        wl_success = wl_eval_result["success"]
        wl_error_msg = wl_eval_result["message"]
    else:
        code_success = False
        code_error_msg = code_eval_result["message"]
        code = code_eval_result["lp_file_path"]
        wl_success = False
        wl_error_msg = ""

    reward = {
        "code_reward": 1.0 if code_success else 0.0,
        "wl_reward": 1.0 if wl_success else 0.0,
    }
    verification = {
        "code_verification": code_error_msg,
        "wl_verification": wl_error_msg,
    }
    return code, reward, verification, equivalence_check_time


def run_sample(
    data_dir: str,
    sample_data: dict,
    model_name: str,
    generation_kwargs: dict,
    eval_timeout_seconds: int,
) -> dict:
    result = dict(sample_data)
    result["prompt"] = generate_prompt(result)
    result["completion"] = generate_completion(result, model_name, **generation_kwargs)

    try:
        if _eval_semaphore is not None:
            with _eval_semaphore:
                code, reward, verification, equivalence_check_time = evaluate_completion_with_timeout(
                    data_dir,
                    result,
                    model_name,
                    eval_timeout_seconds,
                )
        else:
            code, reward, verification, equivalence_check_time = evaluate_completion_with_timeout(
                data_dir,
                result,
                model_name,
                eval_timeout_seconds,
            )
    except Exception as exc:
        code = None
        reward = {"code_reward": 0.0, "wl_reward": 0.0}
        verification = {
            "code_verification": f"exception_or_timeout: {str(exc)}",
            "wl_verification": "",
        }
        equivalence_check_time = ""

    result["code"] = code
    result["reward"] = reward
    result["verification"] = verification
    result["equivalence_check_time"] = equivalence_check_time
    return result


def load_existing_results(save_path: str) -> Dict[str, dict]:
    if not os.path.exists(save_path):
        print(f"结果文件 {save_path} 不存在，将从头开始")
        return {}

    try:
        with open(save_path, "r", encoding="utf-8") as handle:
            results = json.load(handle)
    except Exception as exc:
        print(f"加载结果文件时出错: {exc}")
        return {}

    completed_samples = {}
    for result in results:
        sample_id = result.get("id")
        if sample_id:
            completed_samples[sample_id] = result

    print(f"成功加载 {len(completed_samples)} 个已完成的结果")
    return completed_samples


def save_results_incremental(all_results: List[dict], save_path: str, lock: threading.Lock):
    with lock:
        backup_path = save_path + ".backup"
        try:
            if os.path.exists(save_path):
                os.replace(save_path, backup_path)

            with open(save_path, "w", encoding="utf-8") as handle:
                json.dump(all_results, handle, ensure_ascii=False, indent=4)

            if os.path.exists(backup_path):
                os.remove(backup_path)
            print(f"结果已保存到 {save_path}")
        except Exception as exc:
            print(f"保存结果时出错: {exc}")
            if os.path.exists(backup_path):
                os.replace(backup_path, save_path)


def sort_results_by_numeric_id(results_by_id: Dict[str, dict]) -> List[dict]:
    def sample_sort_key(sample_id: str):
        try:
            return int(sample_id.split("_")[1])
        except (IndexError, ValueError):
            return sample_id

    return [results_by_id[sample_id] for sample_id in sorted(results_by_id.keys(), key=sample_sort_key)]


def filter_pending_samples(full_data: List[dict], completed_sample_ids: Set[str]) -> List[dict]:
    pending_samples = [sample for sample in full_data if sample["id"] not in completed_sample_ids]
    print(f"总样本数: {len(full_data)}, 已完成: {len(completed_sample_ids)}, 待处理: {len(pending_samples)}")
    return pending_samples


def main():
    global _eval_semaphore

    parser = argparse.ArgumentParser(description="Fast resume-capable pipeline evaluation")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Name of the tested model")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--data_dir", type=str, default="data/bench4opt_mix", help="Path to data directory")
    parser.add_argument("--max_workers", type=int, default=min(8, os.cpu_count() or 4), help="Maximum worker threads")
    parser.add_argument("--eval_workers", type=int, default=1, help="Maximum concurrent evaluation workers (recommended: 1)")
    parser.add_argument("--eval_timeout", type=int, default=180, help="Timeout in seconds for each sample evaluation")
    parser.add_argument("--save_every", type=int, default=16, help="Save after every N completed samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for pending samples")
    parser.add_argument("--seed", type=int, default=42, help="Seed forwarded to the model API")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    existing_results = load_existing_results(args.save_path)
    completed_sample_ids = set(existing_results.keys())

    full_data = load_samples(args.data_dir)
    pending_samples = filter_pending_samples(full_data, completed_sample_ids)
    if args.max_samples is not None:
        pending_samples = pending_samples[: args.max_samples]

    if not pending_samples:
        print("所有样本都已完成评测！")
        return

    results_by_id = dict(existing_results)
    save_lock = threading.Lock()
    generation_kwargs = {"seed": args.seed}
    completed_since_last_save = 0
    max_workers = max(1, min(args.max_workers, len(pending_samples)))
    eval_workers = max(1, args.eval_workers)
    _eval_semaphore = threading.Semaphore(eval_workers)

    print(
        f"使用快速评测模式: workers={max_workers}, eval_workers={eval_workers}, "
        f"eval_timeout={args.eval_timeout}s, save_every={args.save_every}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample_id = {
            executor.submit(
                run_sample,
                args.data_dir,
                sample,
                args.model_name,
                generation_kwargs,
                args.eval_timeout,
            ): sample["id"]
            for sample in pending_samples
        }

        for future in tqdm(as_completed(future_to_sample_id), total=len(future_to_sample_id), desc="快速评测"):
            sample_id = future_to_sample_id[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"样本 {sample_id} 处理失败: {exc}")
                continue

            results_by_id[sample_id] = result
            completed_since_last_save += 1

            if completed_since_last_save >= args.save_every:
                save_results_incremental(sort_results_by_numeric_id(results_by_id), args.save_path, save_lock)
                completed_since_last_save = 0

    save_results_incremental(sort_results_by_numeric_id(results_by_id), args.save_path, save_lock)
    print(f"评测完成！总共保存了 {len(results_by_id)} 个样本，结果已保存到 {args.save_path}")


if __name__ == "__main__":
    main()