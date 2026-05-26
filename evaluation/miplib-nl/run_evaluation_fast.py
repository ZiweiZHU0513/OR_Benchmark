"""miplib-nl LP evaluation using the accelerated sufficiency-aware checker."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompt_template_lp_structure.txt"
)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LP_UTILS_PATH = os.path.join(REPO_ROOT, "evaluate", "_lp_utils.py")
EQUIV_CHECK_PATH = os.path.join(REPO_ROOT, "evaluate", "equivalence_check.py")

RESULT_START = "__RESULT_JSON_START__"
RESULT_END = "__RESULT_JSON_END__"

WRAPPER_TEMPLATE = '''import json
import os
import sys
import builtins
import io

os.environ["GRB_LICENSE_FILE"] = os.environ.get("GRB_LICENSE_FILE", "")

try:
    import gurobipy as gp
    from gurobipy import GRB

    instance_dir = {instance_dir_repr}
    data_dir = os.path.join(instance_dir, "data")
    output_lp_path = {output_lp_path_repr}
    user_code = {user_code_repr}
    sentinel = "__MODEL_CAPTURED__"
    captured_models = []

    original_open = builtins.open
    original_io_open = io.open
    original_exists = os.path.exists
    original_isfile = os.path.isfile
    original_optimize = gp.Model.optimize
    original_read = gp.read

    data_files_by_basename = {{}}
    if original_exists(data_dir):
        for root, _, filenames in os.walk(data_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                data_files_by_basename.setdefault(filename, []).append(file_path)

    def resolve_path_alias(path_value):
        if not isinstance(path_value, (str, bytes, os.PathLike)):
            return path_value

        raw_path = os.fspath(path_value)
        if not raw_path or original_exists(raw_path):
            return raw_path

        candidates = []
        basename = os.path.basename(raw_path.rstrip(os.sep))

        if os.path.isabs(raw_path):
            try:
                rel_path = os.path.relpath(raw_path, instance_dir)
            except Exception:
                rel_path = None

            if rel_path and not rel_path.startswith(".."):
                rel_path = rel_path.lstrip("./")
                if rel_path and not rel_path.startswith("data" + os.sep):
                    candidates.append(os.path.join(data_dir, rel_path))
                if basename:
                    candidates.append(os.path.join(data_dir, basename))
        else:
            rel_path = raw_path[2:] if raw_path.startswith("./") else raw_path
            rel_path = rel_path.lstrip("./")
            if rel_path:
                candidates.append(os.path.join(instance_dir, rel_path))
                if not rel_path.startswith("data" + os.sep):
                    candidates.append(os.path.join(data_dir, rel_path))
                if basename:
                    candidates.append(os.path.join(data_dir, basename))

        basename_matches = data_files_by_basename.get(basename, [])
        if len(basename_matches) == 1:
            candidates.append(basename_matches[0])

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if original_exists(candidate):
                return candidate

        return raw_path

    def patched_open(file, mode="r", *args, **kwargs):
        resolved_file = file
        if not any(flag in mode for flag in ("w", "a", "x", "+")):
            resolved_file = resolve_path_alias(file)
        return original_open(resolved_file, mode, *args, **kwargs)

    def patched_io_open(file, mode="r", *args, **kwargs):
        resolved_file = file
        if not any(flag in mode for flag in ("w", "a", "x", "+")):
            resolved_file = resolve_path_alias(file)
        return original_io_open(resolved_file, mode, *args, **kwargs)

    def patched_exists(path_value):
        return original_exists(resolve_path_alias(path_value))

    def patched_isfile(path_value):
        return original_isfile(resolve_path_alias(path_value))

    def patched_read(*args, **kwargs):
        if args:
            args = (resolve_path_alias(args[0]),) + args[1:]
        elif "filename" in kwargs:
            kwargs["filename"] = resolve_path_alias(kwargs["filename"])
        model = original_read(*args, **kwargs)
        captured_models.append(model)
        return model

    def patched_optimize(self, *args, **kwargs):
        captured_models.append(self)
        try:
            self.write(output_lp_path)
        except Exception:
            pass
        raise SystemExit(sentinel)

    builtins.open = patched_open
    io.open = patched_io_open
    os.path.exists = patched_exists
    os.path.isfile = patched_isfile
    gp.Model.optimize = patched_optimize
    gp.read = patched_read

    namespace = {{"gp": gp, "GRB": GRB, "__name__": "__main__"}}
    try:
        exec(user_code, namespace, namespace)
    except SystemExit as exc:
        if str(exc) != sentinel:
            raise
    finally:
        builtins.open = original_open
        io.open = original_io_open
        os.path.exists = original_exists
        os.path.isfile = original_isfile
        gp.Model.optimize = original_optimize
        gp.read = original_read

    model_obj = None
    if captured_models:
        model_obj = captured_models[-1]

    for name in ["model", "m", "opt_model", "lp_model", "mdl"]:
        obj = namespace.get(name)
        if isinstance(obj, gp.Model):
            model_obj = obj
            break

    if model_obj is None:
        for obj in namespace.values():
            if isinstance(obj, gp.Model):
                model_obj = obj
                break

    if model_obj is None:
        raise RuntimeError("MODEL_NOT_FOUND")

    if not os.path.exists(output_lp_path):
        model_obj.write(output_lp_path)

    print("{result_start}")
    print(json.dumps({{"success": True, "message": "LP_WRITTEN"}}))
    print("{result_end}")
except Exception as exc:
    print("{result_start}")
    print(json.dumps({{"success": False, "message": str(exc)}}))
    print("{result_end}")
    sys.exit(1)
'''


def load_module_from_path(module_name: str, module_path: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lp_utils_mod = load_module_from_path("miplib_nl_lp_utils", LP_UTILS_PATH)
extract_python_code = _lp_utils_mod.extract_python_code
_equiv_mod = load_module_from_path("miplib_nl_equivcheck", EQUIV_CHECK_PATH)
check_lp_equivalence_fast = _equiv_mod.check_lp_equivalence_fast


def load_prompt_template() -> str:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def get_gurobi_version() -> str:
    try:
        import gurobipy as gp

        return ".".join(str(part) for part in gp.gurobi.version())
    except Exception:
        return "unknown"


def render_prompt(template: str, instance: dict, instance_dir: str, output_lp_path: str) -> str:
    problem_desc = instance.get("abstract_problem", "")
    params = instance.get("parameters", {})
    files = instance.get("files")

    prompt = template
    prompt = prompt.replace("[Problem Description]", problem_desc)
    prompt = prompt.replace("[Parameters JSON]", json.dumps(params, ensure_ascii=False, indent=2))
    prompt = prompt.replace(
        "[Data Files Info]",
        json.dumps(files, ensure_ascii=False, indent=2)
        if files
        else "No external data files for this problem.",
    )
    prompt = prompt.replace("[Working Directory]", instance_dir)
    prompt = prompt.replace("[LP Output Path]", output_lp_path)
    prompt = prompt.replace("[Gurobi Version]", get_gurobi_version())
    return prompt


def list_instances(dataset_root: str) -> list:
    items = []
    for name in sorted(os.listdir(dataset_root)):
        instance_dir = os.path.join(dataset_root, name)
        instance_json = os.path.join(instance_dir, "instance.json")
        reference_lp_path = os.path.join(instance_dir, f"{name}.lp")
        if not os.path.isdir(instance_dir):
            continue
        if not os.path.isfile(instance_json):
            continue
        items.append(
            {
                "sample_id": name,
                "instance_dir": os.path.abspath(instance_dir),
                "instance_json": os.path.abspath(instance_json),
                "reference_lp_path": os.path.abspath(reference_lp_path),
            }
        )
    return items


def generate_completion(client, messages, model_name, **kwargs):
    from openai import APIError, InternalServerError

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except (InternalServerError, APIError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def parse_wrapper_output(output: str) -> dict:
    start = output.rfind(RESULT_START)
    end = output.rfind(RESULT_END)
    if start == -1 or end == -1 or end <= start:
        return {}
    payload = output[start + len(RESULT_START):end].strip()
    try:
        return json.loads(payload)
    except Exception:
        return {}


def convert_completion_to_lp(
    code: str,
    instance_dir: str,
    output_lp_path: str,
    timeout: int,
) -> tuple:
    extracted_code = extract_python_code(code)

    os.makedirs(os.path.dirname(output_lp_path), exist_ok=True)
    if os.path.exists(output_lp_path):
        os.remove(output_lp_path)

    temp_dir = tempfile.mkdtemp(prefix="miplibnl_lp_")
    try:
        wrapper_code = WRAPPER_TEMPLATE.format(
            instance_dir_repr=repr(instance_dir),
            output_lp_path_repr=repr(output_lp_path),
            user_code_repr=repr(extracted_code),
            result_start=RESULT_START,
            result_end=RESULT_END,
        )
        wrapper_path = os.path.join(temp_dir, "wrapper.py")
        with open(wrapper_path, "w", encoding="utf-8") as handle:
            handle.write(wrapper_code)

        try:
            result = subprocess.run(
                [sys.executable, wrapper_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=instance_dir,
            )
        except subprocess.TimeoutExpired:
            return False, extracted_code, f"LP_BUILD_TIMEOUT: conversion timed out after {timeout}s"
        except Exception as exc:
            return False, extracted_code, f"LP_BUILD_ERROR: subprocess failed: {exc}"

        wrapper_result = parse_wrapper_output((result.stdout or "") + "\n" + (result.stderr or ""))
        if os.path.exists(output_lp_path):
            return True, extracted_code, wrapper_result.get("message", "LP_WRITTEN")

        stderr_tail = " | ".join((result.stderr or "").strip().splitlines()[-5:])
        message = wrapper_result.get("message", "LP file was not created")
        if stderr_tail:
            message = f"{message}; stderr: {stderr_tail[:300]}"
        return False, extracted_code, f"LP_BUILD_ERROR: {message}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def evaluate_one(
    sample: dict,
    client,
    model_name: str,
    prompt_template: str,
    generation_kwargs: dict,
    timeout: int,
    verbose: bool,
) -> dict:
    sample_id = sample["sample_id"]

    try:
        with open(sample["instance_json"], "r", encoding="utf-8") as handle:
            instance = json.load(handle)
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "success": False,
            "msg": f"DATA_ERROR: failed to load instance.json: {exc}",
            "completion": "",
            "extracted_code": "",
            "lp_file_path": "",
            "reference_lp_path": sample["reference_lp_path"],
            "prompt_chars": 0,
            "eval_time": 0.0,
            "api_error": False,
        }

    if not os.path.exists(sample["reference_lp_path"]):
        return {
            "sample_id": sample_id,
            "success": False,
            "msg": "DATA_ERROR: reference LP file not found",
            "completion": "",
            "extracted_code": "",
            "lp_file_path": "",
            "reference_lp_path": sample["reference_lp_path"],
            "prompt_chars": 0,
            "eval_time": 0.0,
            "api_error": False,
        }

    prompt = render_prompt(
        prompt_template,
        instance,
        sample["instance_dir"],
        sample["output_lp_path"],
    )

    try:
        completion = generate_completion(
            client,
            [{"role": "user", "content": prompt}],
            model_name,
            **generation_kwargs,
        )
    except Exception as exc:
        if verbose:
            import traceback

            traceback.print_exc()
        return {
            "sample_id": sample_id,
            "success": False,
            "msg": f"API_ERROR: {str(exc)[:300]}",
            "completion": "",
            "extracted_code": "",
            "lp_file_path": "",
            "reference_lp_path": sample["reference_lp_path"],
            "prompt_chars": len(prompt),
            "eval_time": 0.0,
            "api_error": True,
        }

    eval_start = time.time()
    lp_success, extracted_code, build_message = convert_completion_to_lp(
        completion,
        sample["instance_dir"],
        sample["output_lp_path"],
        timeout,
    )
    if not lp_success:
        return {
            "sample_id": sample_id,
            "success": False,
            "msg": build_message,
            "completion": completion,
            "extracted_code": extracted_code,
            "lp_file_path": sample["output_lp_path"],
            "reference_lp_path": sample["reference_lp_path"],
            "prompt_chars": len(prompt),
            "eval_time": time.time() - eval_start,
            "api_error": False,
            "equivalence_check_time": {},
        }

    equivalence_success, equivalence_msg, time_info = check_lp_equivalence_fast(
        sample["reference_lp_path"],
        sample["output_lp_path"],
    )
    eval_time = time.time() - eval_start

    return {
        "sample_id": sample_id,
        "success": equivalence_success,
        "msg": equivalence_msg,
        "completion": completion,
        "extracted_code": extracted_code,
        "lp_file_path": sample["output_lp_path"],
        "reference_lp_path": sample["reference_lp_path"],
        "prompt_chars": len(prompt),
        "eval_time": eval_time,
        "api_error": False,
        "equivalence_check_time": time_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data/miplib-nl")
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
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--save_path", default="results/miplib-nl/gpt-4o_fast.json")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--verbose", type=lambda x: str(x).lower() == "true", default=False)
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=2))

    base_model_name = args.model_name.replace("/", "-")
    lp_output_dir = os.path.abspath(os.path.join("temp_lp", f"{base_model_name}_miplib_nl_fast"))
    os.makedirs(lp_output_dir, exist_ok=True)

    samples = list_instances(args.dataset_root)
    end = args.end if args.end is not None else len(samples)
    samples = samples[args.start:end]
    for sample in samples:
        sample["output_lp_path"] = os.path.join(lp_output_dir, f"{sample['sample_id']}.lp")

    output_data = []
    completed_ids = set()

    if args.save_path and os.path.exists(args.save_path) and not args.rerun:
        try:
            with open(args.save_path, "r", encoding="utf-8") as handle:
                output_data = json.load(handle)
            for item in output_data:
                if not item.get("api_error", False) and "success" in item:
                    completed_ids.add(item["sample_id"])
            output_data = [item for item in output_data if item.get("sample_id") in completed_ids]
            print(f"Resume mode: {len(completed_ids)} already done, api_error entries will be retried")
        except Exception as exc:
            print(f"Warning: could not load existing results: {exc}")
            output_data = []
            completed_ids = set()
    elif args.rerun:
        print("Rerun mode: processing all samples from scratch")

    to_process = [sample for sample in samples if sample["sample_id"] not in completed_ids]
    print(f"To process: {len(to_process)} / {len(samples)}")

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    prompt_template = load_prompt_template()

    results = []
    lock = threading.Lock()

    def worker(sample):
        result = evaluate_one(
            sample,
            client,
            args.model_name,
            prompt_template,
            generation_kwargs,
            args.timeout,
            args.verbose,
        )
        with lock:
            results.append(result)
            if args.save_path:
                os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                tmp_path = args.save_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    json.dump(output_data + results, handle, ensure_ascii=False, indent=2)
                os.replace(tmp_path, args.save_path)
        return result

    if to_process:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(worker, sample): sample for sample in to_process}
            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Evaluating {args.model_name}",
                dynamic_ncols=True,
            )
            done = 0
            success = 0
            for future in progress:
                done += 1
                try:
                    result = future.result()
                    if result.get("success"):
                        success += 1
                except Exception as exc:
                    if args.verbose:
                        print(f"Worker error: {exc}")
                progress.set_postfix(acc=f"{success}/{done} ({success / done * 100:.1f}%)")

    final_results = output_data + results
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as handle:
            json.dump(final_results, handle, ensure_ascii=False, indent=2)
        print(f"Saved to {args.save_path}")

    total = len(final_results)
    successful = sum(1 for item in final_results if item.get("success"))
    api_errors = sum(1 for item in final_results if item.get("api_error", False))
    print(f"Total samples: {total}")
    print(f"Successful: {successful}")
    print(f"API errors: {api_errors}")
    print(f"Accuracy: {successful / total * 100:.2f}%" if total else "Accuracy: 0.00%")


if __name__ == "__main__":
    main()