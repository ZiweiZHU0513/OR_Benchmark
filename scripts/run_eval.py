#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET_ORDER = [
    "nl4opt_solver",
    "optibench_solver",
    "miplib_solver",
    "miplib_orgeval",
    "bench4opt_orgeval",
]

VALID_SOLVER_STATUSES = {"OPTIMAL", "INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize evaluation results produced by scripts/run_eval_suite.sh"
    )
    parser.add_argument("--models", nargs="+", required=True, help="Model names to summarize")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGET_ORDER,
        default=list(TARGET_ORDER),
        help="Subset of result groups to summarize",
    )
    parser.add_argument("--result_root", default="results", help="Root directory for raw evaluation outputs")
    parser.add_argument(
        "--summary_root",
        default=None,
        help="Summary root directory, default: <result_root>/summary",
    )
    parser.add_argument("--summary_tag", default=None, help="Optional summary subdirectory name")
    parser.add_argument("--start", type=int, default=0, help="Shared start index for sliced outputs")
    parser.add_argument("--end", type=int, default=None, help="Shared end index for sliced outputs")
    parser.add_argument(
        "--bench4opt_max_samples",
        type=int,
        default=None,
        help="Optional max_samples suffix for bench4opt orgeval outputs",
    )
    parser.add_argument("--nl4opt_dataset", default="./data/NL4OPT")
    parser.add_argument("--optibench_dataset", default="./data/optibench")
    parser.add_argument("--miplib_solver_dataset", default="data/miplib-nl")
    parser.add_argument("--miplib_orgeval_dataset", default="data/miplib-nl_exclude_failure")
    parser.add_argument("--bench4opt_dataset", default="data/bench4opt")
    parser.add_argument("--skip_missing_results", action="store_true")
    parser.add_argument("--skip_run", action="store_true")
    return parser.parse_args()


def safe_model_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", value)


def range_suffix(args: argparse.Namespace) -> str:
    if args.start != 0 or args.end is not None:
        end_text = "none" if args.end is None else str(args.end)
        return f"_{args.start}_{end_text}"
    return ""


def target_display_name(target_key: str, args: argparse.Namespace) -> str:
    if target_key == "nl4opt_solver":
        return "nl4opt"
    if target_key == "optibench_solver":
        return "optibench"
    if target_key == "miplib_solver":
        return "miplib-nl"
    if target_key == "miplib_orgeval":
        return Path(args.miplib_orgeval_dataset).name.replace("_", "-")
    if target_key == "bench4opt_orgeval":
        return "bench4opt"
    raise KeyError(f"Unknown target: {target_key}")


def result_path_for_target(target_key: str, model_name: str, args: argparse.Namespace) -> Path:
    result_root = (REPO_ROOT / args.result_root).resolve()
    safe_model = safe_model_name(model_name)
    slice_suffix = range_suffix(args)

    if target_key == "nl4opt_solver":
        return result_root / "nl4opt" / f"{safe_model}{slice_suffix}_solver.json"
    if target_key == "optibench_solver":
        return result_root / "optibench" / f"{safe_model}{slice_suffix}_solver.json"
    if target_key == "miplib_solver":
        return result_root / "miplib-nl" / f"{safe_model}{slice_suffix}_solver.json"
    if target_key == "miplib_orgeval":
        return result_root / "miplib-nl" / f"{safe_model}{slice_suffix}_orgeval.json"
    if target_key == "bench4opt_orgeval":
        max_suffix = f"_max{args.bench4opt_max_samples}" if args.bench4opt_max_samples is not None else ""
        return result_root / "bench4opt" / f"{safe_model}{max_suffix}_orgeval.json"
    raise KeyError(f"Unknown target: {target_key}")


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON result file: {path}")
    return payload


def solver_compile_pass(target_key: str, item: Dict[str, Any]) -> bool:
    if target_key == "miplib_solver":
        predicted_status = str(item.get("predicted_status") or "").strip().upper()
        return item.get("predicted_optimal_value") is not None or predicted_status in VALID_SOLVER_STATUSES
    return item.get("predicted_optimal_value") is not None


def bench4opt_code_pass(item: Dict[str, Any]) -> bool:
    reward = item.get("reward") or {}
    try:
        return float(reward.get("code_reward", 0.0)) >= 1.0
    except (TypeError, ValueError):
        return False


def bench4opt_success(item: Dict[str, Any]) -> bool:
    reward = item.get("reward") or {}
    try:
        return float(reward.get("wl_reward", 0.0)) >= 1.0
    except (TypeError, ValueError):
        return False


def miplib_orgeval_code_pass(item: Dict[str, Any]) -> bool:
    if item.get("api_error", False):
        return False
    if item.get("success"):
        return True
    message = item.get("msg")
    if isinstance(message, dict):
        return True
    text = str(message or "")
    return not text.startswith(("LP_BUILD_", "DATA_ERROR", "API_ERROR"))


def classify_build_error(payload: Any) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    lower = text.lower()

    if "api_error" in lower:
        return "api_error"
    if "timeout" in lower:
        return "timeout"
    if "no such file or directory" in lower:
        return "missing_file_or_path"
    if "error tokenizing data" in lower or "no columns to parse" in lower or "unexpected number of columns" in lower:
        return "data_parse_error"
    if "syntaxerror" in lower or "positional argument follows keyword argument" in lower:
        return "syntax_error"
    if "model_not_found" in lower:
        return "model_not_found"
    if "nameerror" in lower or "is not defined" in lower:
        return "name_error"
    if "keyerror" in lower:
        return "key_error"
    if "indexerror" in lower or "out of bounds" in lower:
        return "index_error"
    if (
        "typeerror" in lower
        or "unsupported operand type" in lower
        or "unhashable type" in lower
        or "not subscriptable" in lower
        or "can't multiply sequence" in lower
        or "must be real number" in lower
    ):
        return "type_error"
    if "valueerror" in lower or "could not convert" in lower or "invalid literal" in lower or "length mismatch" in lower:
        return "value_error"
    if "addconstr" in lower or "addvars" in lower or "duplicate keys in model.addvars" in lower:
        return "gurobi_api_error"
    if "lp_build_error" in lower or "lp file was not created" in lower:
        return "lp_build_failure"
    return "other_build_error"


def classify_equivalence_error(payload: Any) -> str:
    if isinstance(payload, dict):
        if payload.get("var_num_check") is False:
            return "var_count_mismatch"
        if payload.get("cons_num_check") is False:
            return "constraint_count_mismatch"
        if payload.get("wl_check") is False:
            return "wl_graph_mismatch"
        false_keys = [key for key, value in payload.items() if value is False]
        if false_keys:
            return "structure_mismatch:" + ",".join(sorted(false_keys))
        return "other_equivalence_error"

    text = str(payload or "")
    lower = text.lower()
    if "no reference lp" in lower:
        return "missing_reference_lp"
    if "normalization" in lower:
        return "normalization_error"
    return "other_equivalence_error"


def classify_orgeval_failure(target_key: str, item: Dict[str, Any]) -> str:
    if item.get("api_error", False):
        return "api_error"

    if target_key == "bench4opt_orgeval":
        if bench4opt_success(item):
            return "correct"
        if not bench4opt_code_pass(item):
            verification = item.get("verification") or {}
            return classify_build_error(verification.get("code_verification", ""))
        verification = item.get("verification") or {}
        return classify_equivalence_error(verification.get("wl_verification", ""))

    if item.get("success"):
        return "correct"
    if not miplib_orgeval_code_pass(item):
        return classify_build_error(item.get("msg", ""))
    return classify_equivalence_error(item.get("msg", ""))


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_solver_summary(result_map: Dict[str, Dict[str, List[Dict[str, Any]]]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target_key in ("nl4opt_solver", "optibench_solver", "miplib_solver"):
        if target_key not in result_map:
            continue
        dataset_name = target_display_name(target_key, args)
        for model_name in args.models:
            items = result_map[target_key].get(model_name)
            if items is None:
                continue
            total = len(items)
            success_count = sum(1 for item in items if item.get("success"))
            compile_count = sum(1 for item in items if solver_compile_pass(target_key, item))
            api_error_count = sum(1 for item in items if item.get("api_error", False))
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "accuracy": ratio(success_count, total),
                    "accuracy_count": success_count,
                    "compile_pass_rate": ratio(compile_count, total),
                    "compile_pass_count": compile_count,
                    "api_error_count": api_error_count,
                    "total": total,
                }
            )
    return rows


def build_orgeval_summary(result_map: Dict[str, Dict[str, List[Dict[str, Any]]]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target_key in ("miplib_orgeval", "bench4opt_orgeval"):
        if target_key not in result_map:
            continue
        dataset_name = target_display_name(target_key, args)
        for model_name in args.models:
            items = result_map[target_key].get(model_name)
            if items is None:
                continue
            total = len(items)
            if target_key == "bench4opt_orgeval":
                success_count = sum(1 for item in items if bench4opt_success(item))
                code_count = sum(1 for item in items if bench4opt_code_pass(item))
            else:
                success_count = sum(1 for item in items if item.get("success"))
                code_count = sum(1 for item in items if miplib_orgeval_code_pass(item))
            api_error_count = sum(1 for item in items if item.get("api_error", False))
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "accuracy": ratio(success_count, total),
                    "accuracy_count": success_count,
                    "compile_pass_rate": ratio(code_count, total),
                    "compile_pass_count": code_count,
                    "api_error_count": api_error_count,
                    "total": total,
                }
            )
    return rows


def build_orgeval_error_summary(result_map: Dict[str, Dict[str, List[Dict[str, Any]]]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target_key in ("miplib_orgeval", "bench4opt_orgeval"):
        if target_key not in result_map:
            continue
        dataset_name = target_display_name(target_key, args)
        for model_name in args.models:
            items = result_map[target_key].get(model_name)
            if items is None:
                continue
            failures = [item for item in items if classify_orgeval_failure(target_key, item) != "correct"]
            counter = Counter(classify_orgeval_failure(target_key, item) for item in failures)
            total = len(items)
            failed_total = len(failures)
            for error_type, count in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0])):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "error_type": error_type,
                        "count": count,
                        "share_of_total": ratio(count, total),
                        "share_of_failures": ratio(count, failed_total),
                        "total": total,
                        "failed_total": failed_total,
                    }
                )
    return rows


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    collected_rows = list(rows)
    if not collected_rows:
        return "无数据"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in collected_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(
    solver_rows: List[Dict[str, Any]],
    orgeval_rows: List[Dict[str, Any]],
    error_rows: List[Dict[str, Any]],
    result_paths: Dict[str, Dict[str, str]],
) -> str:
    solver_table = markdown_table(
        ["数据集", "模型", "求解正确率", "Solver代码compile通过率", "成功数", "Compile通过数", "总样本", "API错误"],
        [
            [
                row["dataset"],
                row["model"],
                format_percent(row["accuracy"]),
                format_percent(row["compile_pass_rate"]),
                str(row["accuracy_count"]),
                str(row["compile_pass_count"]),
                str(row["total"]),
                str(row["api_error_count"]),
            ]
            for row in solver_rows
        ],
    )
    orgeval_table = markdown_table(
        ["数据集", "模型", "图同构正确率", "LP构建通过率", "正确数", "构建通过数", "总样本", "API错误"],
        [
            [
                row["dataset"],
                row["model"],
                format_percent(row["accuracy"]),
                format_percent(row["compile_pass_rate"]),
                str(row["accuracy_count"]),
                str(row["compile_pass_count"]),
                str(row["total"]),
                str(row["api_error_count"]),
            ]
            for row in orgeval_rows
        ],
    )
    error_table = markdown_table(
        ["数据集", "模型", "错误类型", "数量", "占总体比例", "占失败样本比例"],
        [
            [
                row["dataset"],
                row["model"],
                row["error_type"],
                str(row["count"]),
                format_percent(row["share_of_total"]),
                format_percent(row["share_of_failures"]),
            ]
            for row in error_rows
        ],
    )
    raw_paths = []
    for target_key, model_map in result_paths.items():
        for model_name, path in model_map.items():
            raw_paths.append(f"- {target_key} / {model_name}: {path}")

    sections = [
        f"# Evaluation Summary\n\nGenerated at: {datetime.now().isoformat(timespec='seconds')}",
        "## Solver 测评结果表\n\n" + solver_table,
        "## Orgeval 测评结果表\n\n" + orgeval_table,
        "## Orgeval 错误分析表\n\n" + error_table,
        "## 原始结果文件\n\n" + ("\n".join(raw_paths) if raw_paths else "- 无结果路径"),
    ]
    return "\n\n".join(sections)


def summary_directory(args: argparse.Namespace) -> Path:
    summary_root = args.summary_root or str(Path(args.result_root) / "summary")
    summary_tag = args.summary_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / summary_root / summary_tag).resolve()


def load_all_results(args: argparse.Namespace) -> tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Dict[str, str]]]:
    result_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    result_paths: Dict[str, Dict[str, str]] = {}

    for target_key in args.targets:
        result_map[target_key] = {}
        result_paths[target_key] = {}
        for model_name in args.models:
            path = result_path_for_target(target_key, model_name, args)
            result_paths[target_key][model_name] = str(path)
            if not path.exists():
                if args.skip_missing_results:
                    print(f"[WARN] skip missing result: {path}")
                    continue
                raise FileNotFoundError(f"Missing result file: {path}")
            result_map[target_key][model_name] = load_results(path)

    return result_map, result_paths


def main() -> None:
    args = parse_args()
    result_map, result_paths = load_all_results(args)

    solver_rows = build_solver_summary(result_map, args)
    orgeval_rows = build_orgeval_summary(result_map, args)
    error_rows = build_orgeval_error_summary(result_map, args)

    summary_dir = summary_directory(args)
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = summary_dir / "summary.json"
    summary_md_path = summary_dir / "summary.md"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "result_paths": result_paths,
        "solver_summary": solver_rows,
        "orgeval_summary": orgeval_rows,
        "orgeval_error_summary": error_rows,
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    summary_markdown = render_markdown(solver_rows, orgeval_rows, error_rows, result_paths)
    with summary_md_path.open("w", encoding="utf-8") as handle:
        handle.write(summary_markdown + "\n")

    print("\n=== Summary written ===")
    print(summary_json_path)
    print(summary_md_path)
    print("\n=== Summary preview ===")
    print(summary_markdown)


if __name__ == "__main__":
    main()