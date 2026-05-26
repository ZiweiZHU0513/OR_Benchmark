"""
Solver-based evaluation for miplib-nl benchmark.
Executes generated gurobi code inside the instance directory (so ./data/*.csv reads work),
then compares model.ObjVal against the GT optimal_value.
"""
import os
import sys
import json
import shutil
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, Any

from evaluation.bench4opt.utils.lp_utils import extract_python_code


WRAPPER_TEMPLATE = '''import sys
import json
import os

os.environ['GRB_LICENSE_FILE'] = os.environ.get('GRB_LICENSE_FILE', '')

try:
    import gurobipy as gp
    from gurobipy import GRB

    user_code = {user_code_repr}
    _ns = {{}}
    exec(user_code, _ns, _ns)

    model_obj = None
    for var_name in ['model', 'm', 'opt_model', 'mip', 'mdl']:
        if var_name in _ns:
            obj = _ns[var_name]
            if hasattr(obj, 'optimize') and hasattr(obj, 'ObjVal'):
                model_obj = obj
                break

    if model_obj is None:
        print(json.dumps({{"status": -1, "optimal_value": None, "status_name": "MODEL_NOT_FOUND"}}))
        sys.exit(1)

    try:
        model_obj.setParam("OutputFlag", 0)
    except Exception:
        pass

    try:
        model_obj.setParam("Threads", 1)
    except Exception:
        pass

    try:
        if not hasattr(model_obj, 'status') or model_obj.status == GRB.LOADED:
            model_obj.optimize()
        if model_obj.status == GRB.OPTIMAL:
            result = {{"status": 2, "optimal_value": model_obj.ObjVal, "status_name": "OPTIMAL"}}
        elif model_obj.status == GRB.SUBOPTIMAL:
            result = {{"status": int(model_obj.status), "optimal_value": model_obj.ObjVal, "status_name": "SUBOPTIMAL"}}
        elif model_obj.status == GRB.TIME_LIMIT and getattr(model_obj, "SolCount", 0) > 0:
            result = {{"status": int(model_obj.status), "optimal_value": model_obj.ObjVal, "status_name": "TIME_LIMIT_WITH_SOLUTION"}}
        elif model_obj.status == GRB.INFEASIBLE:
            result = {{"status": 3, "optimal_value": None, "status_name": "INFEASIBLE"}}
        elif model_obj.status == GRB.UNBOUNDED:
            result = {{"status": 5, "optimal_value": None, "status_name": "UNBOUNDED"}}
        elif model_obj.status == GRB.INF_OR_UNBD:
            result = {{"status": 4, "optimal_value": None, "status_name": "INF_OR_UNBD"}}
        else:
            result = {{"status": int(model_obj.status), "optimal_value": None, "status_name": "STATUS_" + str(model_obj.status)}}
        print("__RESULT_JSON_START__")
        print(json.dumps(result))
        print("__RESULT_JSON_END__")
    except Exception as e:
        print("__RESULT_JSON_START__")
        print(json.dumps({{"status": -1, "optimal_value": None, "status_name": "ERROR: " + str(e)}}))
        print("__RESULT_JSON_END__")
        sys.exit(1)
except Exception as e:
    import traceback
    print("__RESULT_JSON_START__")
    print(json.dumps({{"status": -1, "optimal_value": None, "status_name": "ERROR: " + str(e)}}))
    print("__RESULT_JSON_END__")
    sys.exit(1)
'''


def _parse_wrapped_output(output: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON result emitted by the wrapper between markers."""
    start_tag = "__RESULT_JSON_START__"
    end_tag = "__RESULT_JSON_END__"
    s = output.rfind(start_tag)
    e = output.rfind(end_tag)
    if s == -1 or e == -1 or e <= s:
        # fallback: try parsing the last JSON object in output
        last_brace = output.rfind('{"status":')
        if last_brace == -1:
            return None
        try:
            return json.loads(output[last_brace:].splitlines()[0])
        except Exception:
            return None
    payload = output[s + len(start_tag):e].strip()
    try:
        return json.loads(payload)
    except Exception:
        return None


def run_generated_code(
    code: str,
    instance_dir: str,
    timeout: int = 600,
) -> Tuple[Optional[float], str, str]:
    """
    Execute the generated gurobi python code with cwd=instance_dir.
    Returns (optimal_value, status_name, message).
    status_name is one of: OPTIMAL, SUBOPTIMAL, TIME_LIMIT_WITH_SOLUTION,
    INFEASIBLE, UNBOUNDED, INF_OR_UNBD, MODEL_NOT_FOUND, ERROR:..., STATUS_<n>, PARSE_ERROR.
    """
    tmp_dir = tempfile.mkdtemp(prefix="miplibnl_")
    try:
        wrapper_code = WRAPPER_TEMPLATE.format(user_code_repr=repr(code))
        code_file = os.path.join(tmp_dir, "model_code.py")
        with open(code_file, "w") as f:
            f.write(wrapper_code)

        try:
            proc = subprocess.run(
                [sys.executable, code_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=instance_dir,
            )
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT", f"Code execution timeout ({timeout}s)"
        except Exception as e:
            return None, "ERROR", f"Subprocess error: {e}"

        output = proc.stdout or ""
        result = _parse_wrapped_output(output)
        if result is None:
            stderr_tail = (proc.stderr or "").strip().splitlines()[-5:]
            return None, "PARSE_ERROR", f"Could not parse output. stderr tail: {' | '.join(stderr_tail)[:300]}"

        status = result.get("status")
        optimal_value = result.get("optimal_value")
        status_name = result.get("status_name", "UNKNOWN")

        if optimal_value is not None and status == 2:
            return optimal_value, status_name, "Optimal solution found"
        if optimal_value is not None:
            return optimal_value, status_name, f"Solution found ({status_name})"
        return None, status_name, f"Optimization failed: {status_name}"
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


_STATUS_KEYWORDS = {
    "infeasible": {"INFEASIBLE", "INF_OR_UNBD"},
    "impossible": {"INFEASIBLE", "INF_OR_UNBD"},
    "unbounded": {"UNBOUNDED", "INF_OR_UNBD"},
}


def evaluate_instance(
    code: str,
    gt_optimal_value,
    instance_dir: str,
    tolerance: float = 1e-4,
    rel_tolerance: float = 1e-3,
    timeout: int = 600,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a model-generated completion against the GT optimal value.

    GT may be:
      - a numeric value: success if abs/rel error within tolerance
      - a string like 'infeasible' / 'unbounded' / 'impossible': success if the
        executed model reports a matching status
    """
    extracted_code = extract_python_code(code)
    if verbose:
        print(f"Extracted code length: {len(extracted_code)}")

    predicted, status_name, message = run_generated_code(extracted_code, instance_dir, timeout=timeout)

    result = {
        "predicted_optimal_value": predicted,
        "predicted_status": status_name,
        "gt_optimal_value": gt_optimal_value,
        "message": message,
    }

    # Categorical GT (infeasible/unbounded/impossible)
    if isinstance(gt_optimal_value, str):
        key = gt_optimal_value.strip().lower()
        expected = _STATUS_KEYWORDS.get(key)
        if expected is None:
            result["success"] = False
            result["error"] = None
            result["message"] = f"Unrecognized categorical GT: {gt_optimal_value!r}"
            return result
        result["success"] = status_name in expected
        result["error"] = None
        if verbose:
            print(f"Categorical GT={key}  predicted_status={status_name}  success={result['success']}")
        return result

    # Numeric GT
    if predicted is None:
        result["success"] = False
        result["error"] = None
        return result

    abs_err = abs(predicted - gt_optimal_value)
    rel_err = abs_err / max(abs(gt_optimal_value), 1e-12)
    result["error"] = abs_err
    result["rel_error"] = rel_err
    result["success"] = (abs_err <= tolerance) or (rel_err <= rel_tolerance)

    if verbose:
        print(f"Predicted: {predicted}  GT: {gt_optimal_value}  abs_err={abs_err}  rel_err={rel_err}  success={result['success']}")

    return result
