from typing import Dict, Any, Union, Optional
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path

# 修改导入语句
from evaluation.bench4opt.utils.lp_utils import extract_python_code
from evaluation.bench4opt.utils.evaluate_code import evaluate_code


def ensure_json_loaded(data):
    """Helper function to ensure data is properly JSON loaded"""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data


def compute_score(
    solution_str: str,
    ground_truth: Dict[str, Any],
    return_dict: bool = False,
    ensure_imports: bool = False,
    save_result: bool = False,
    save_path: Optional[str] = None,
    model_name: Optional[str] = None,
    return_validation_info: bool = False,
    verbose: bool = False,
) -> Union[float, Dict[str, Any]]:
    """Reward function for Bench4Opt.

    Args:
        solution_str: the completion of the model
        ground_truth: the ground truth of the problem
            {
                "sample_id": int,
                "data_source": str, # bench4opt or other
                "data": dict,
                "reference_lp": str,
            }
        save_path: the path to save the lp code of the solution
        return_dict: whether to return the score as a dictionary
        save_result: whether to save the result as a json file
        problem_id: the id of the problem
        model_name: the name of the evaluated model
        verbose: whether to print the score

    Returns:
        the score of the solution
        if return_dict is True, return the score as a dictionary
        {
            "score": float,
            "code_reward": float,
            "wl_reward": float,
        }
        else return the score as a float
    """
    if verbose:
        print(
            f'------------evaluation for problem {ground_truth["sample_id"]}------------'
        )
    code_str = extract_python_code(solution_str)
    code_eval_result, wl_eval_result, equivalence_check_time = evaluate_code(
        data=ensure_json_loaded(ground_truth["data"]),
        code=code_str,
        reference_lp=ground_truth["reference_lp"],
        save_path=save_path,
        model_name=model_name,
        ensure_imports=ensure_imports,
        problem_id=ground_truth["sample_id"],
        data_source=ground_truth["data_source"],
        verbose=verbose,
    )
    code_reward = 1.0 if code_eval_result["success"] else 0.0
    wl_reward = 1.0 if wl_eval_result["success"] else 0.0
    score = {
        "score": code_reward + wl_reward,
        "code_reward": code_reward,
        "wl_reward": wl_reward,
    }
    validation_info = {
        "code_validation": code_eval_result["message"],
        "wl_validation": wl_eval_result["message"],
    }
    if verbose:
        print(f"score: {score}")
        print(f"Code evaluation result: {code_reward}")
        print(f"WL evaluation result: {wl_eval_result}")
        print(f"Equivalence check time: {equivalence_check_time}")
        print(
            f'validation information: {code_eval_result["message"]}; {wl_eval_result["message"]}'
        )
    if save_result:
        result = {
            "sample_id": ground_truth["sample_id"],
            "reward": score,
            "validation": validation_info,
            "equivalence_check_time": equivalence_check_time,
        }
        with open(
            f"bench4opt/outputs/{model_name}_{ground_truth['data_source']}.json", "a"
        ) as f:
            json.dump(result, f, indent=4)
    if return_dict:
        if return_validation_info:
            return score, validation_info
        else:
            return score
    else:
        if return_validation_info:
            return score["score"], validation_info
        else:
            return score["score"]



def test_compute_score():
    solution_str = """
import gurobipy as gp
from gurobipy import GRB
import json

# Load data
with open('data.json', 'r') as file:
    data = json.load(file)

nodes = data['nodes']
edges = data['edges']
capacities = data['capacities']
latencies = data['latencies']
source = data['source']
sink = data['sink']

# Create a new model
model = gp.Model('telecommunications_network')

# Decision variables: flow on each edge
flow = model.addVars(len(edges), lb=0, ub=capacities, name="flow")

# Objective: Minimize total latency
model.setObjective(gp.quicksum(latencies[i] * flow[i] for i in range(len(edges))), GRB.MINIMIZE)

# Constraints: Capacity Constraints
for i in range(len(edges)):
    model.addConstr(flow[i] <= capacities[i], f"capacity_{i}")

# Constraints: Flow Conservation
for node in nodes:
    if node == source:
        model.addConstr(gp.quicksum(flow[i] for i in range(len(edges)) if edges[i][0] == node) - 
                        gp.quicksum(flow[i] for i in range(len(edges)) if edges[i][1] == node) == 1, 
                        f"flow_conservation_{node}")
    elif node == sink:
        model.addConstr(gp.quicksum(flow[i] for i in range(len(edges)) if edges[i][1] == node) - 
                        gp.quicksum(flow[i] for i in range(len(edges)) if edges[i][0] == node) == 1, 
                        f"flow_conservation_{node}")
    else:
        model.addConstr(gp.quicksum(flow[i] for i in range(len(edges)) if edges[i][1] == node) == 
                        gp.quicksum(flow[i] for i in range(len(edges)) if edges[i][0] == node), 
                        f"flow_conservation_{node}")

# Optimize the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print('Optimal Flow:')
    for i in range(len(edges)):
        print(f"Edge {edges[i]}: {flow[i].X}")

# Save model
model.write('network_flow.lp')
"""
    ground_truth = {
        "sample_id": 0,
        "data_source": "bench4opt",
        "data": {},
        "reference_lp": "",
    }
    data_string = """
    {
    "nodes": [
        "Node_0",
        "Node_1",
        "Node_2",
        "Node_3",
        "Node_4",
        "Node_5",
        "Node_6",
        "Node_7",
        "Node_8",
        "Node_9"
    ],
    "edges": [
        [
            "Node_6",
            "Node_7"
        ],
        [
            "Node_4",
            "Node_0"
        ],
        [
            "Node_2",
            "Node_0"
        ],
        [
            "Node_9",
            "Node_3"
        ],
        [
            "Node_7",
            "Node_5"
        ],
        [
            "Node_5",
            "Node_2"
        ],
        [
            "Node_4",
            "Node_6"
        ],
        [
            "Node_4",
            "Node_8"
        ],
        [
            "Node_8",
            "Node_7"
        ],
        [
            "Node_1",
            "Node_6"
        ],
        [
            "Node_5",
            "Node_9"
        ],
        [
            "Node_1",
            "Node_9"
        ],
        [
            "Node_2",
            "Node_4"
        ],
        [
            "Node_6",
            "Node_9"
        ],
        [
            "Node_8",
            "Node_0"
        ],
        [
            "Node_3",
            "Node_5"
        ],
        [
            "Node_3",
            "Node_2"
        ],
        [
            "Node_0",
            "Node_4"
        ],
        [
            "Node_3",
            "Node_6"
        ],
        [
            "Node_9",
            "Node_1"
        ]
    ],
    "capacities": [
        33.220098938197275,
        57.02652991366112,
        66.7542819065456,
        82.2927010977157,
        18.33443580907524,
        39.32948419957436,
        19.390447062527706,
        64.90350948321407,
        17.693093980800437,
        45.02758538137671,
        17.819139969009687,
        82.97529528947074,
        64.21198938160808,
        58.516174836979786,
        32.44083754341121,
        16.812139617079673,
        19.888229035073987,
        35.57449547483052,
        76.24083937750743,
        52.933940637313285
    ],
    "latencies": [
        1.5530343801663886,
        6.030132450340928,
        1.34954649839002,
        0.6765578229065563,
        8.287471212794381,
        1.4208612067220865,
        1.0607150225072361,
        9.7896795888487,
        3.7795861563225,
        6.880024679520408,
        7.010370088899773,
        8.237035261465852,
        7.114416177959136,
        0.41308663280316016,
        3.5772979736541286,
        6.785158346478007,
        4.0156605740295515,
        4.863931303158201,
        2.319471146932837,
        1.0456734995518826
    ],
    "source": "Node_3",
    "sink": "Node_9"
}
    """
    ground_truth["data"] = json.loads(data_string)
    ground_truth[
        "reference_lp"
    ] = """
    \ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  1.553034380166389 f[0] + 6.030132450340928 f[1] + 1.34954649839002 f[2]
   + 0.6765578229065563 f[3] + 8.287471212794381 f[4]
   + 1.420861206722086 f[5] + 1.060715022507236 f[6]
   + 9.789679588848699 f[7] + 3.7795861563225 f[8] + 6.880024679520408 f[9]
   + 7.010370088899773 f[10] + 8.237035261465852 f[11]
   + 7.114416177959136 f[12] + 0.4130866328031602 f[13]
   + 3.577297973654129 f[14] + 6.785158346478007 f[15]
   + 4.015660574029551 f[16] + 4.863931303158201 f[17]
   + 2.319471146932837 f[18] + 1.045673499551883 f[19]
Subject To
 cap_0: f[0] <= 33.22009893819727
 cap_1: f[1] <= 57.02652991366112
 cap_2: f[2] <= 66.7542819065456
 cap_3: f[3] <= 82.29270109771571
 cap_4: f[4] <= 18.33443580907524
 cap_5: f[5] <= 39.32948419957436
 cap_6: f[6] <= 19.39044706252771
 cap_7: f[7] <= 64.90350948321407
 cap_8: f[8] <= 17.69309398080044
 cap_9: f[9] <= 45.02758538137671
 cap_10: f[10] <= 17.81913996900969
 cap_11: f[11] <= 82.97529528947074
 cap_12: f[12] <= 64.21198938160808
 cap_13: f[13] <= 58.51617483697979
 cap_14: f[14] <= 32.44083754341121
 cap_15: f[15] <= 16.81213961707967
 cap_16: f[16] <= 19.88822903507399
 cap_17: f[17] <= 35.57449547483052
 cap_18: f[18] <= 76.24083937750743
 cap_19: f[19] <= 52.93394063731328
 flow_cons_Node_0: f[1] + f[2] + f[14] - f[17] = 0
 flow_cons_Node_1: - f[9] - f[11] + f[19] = 0
 flow_cons_Node_2: - f[2] + f[5] - f[12] + f[16] = 0
 flow_cons_Node_4: - f[1] - f[6] - f[7] + f[12] + f[17] = 0
 flow_cons_Node_5: f[4] - f[5] - f[10] + f[15] = 0
 flow_cons_Node_6: - f[0] + f[6] + f[9] - f[13] + f[18] = 0
 flow_cons_Node_7: f[0] - f[4] + f[8] = 0
 flow_cons_Node_8: f[7] - f[8] - f[14] = 0
Bounds
End
    """
    model_name = "gpt-4o"
    return_dict = True
    ensure_imports = False
    save_result = True
    save_path = "bench4opt/outputs"
    verbose = True
    score = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        return_dict=return_dict,
        ensure_imports=ensure_imports,
        save_result=save_result,
        save_path=save_path,
        model_name=model_name,
        verbose=verbose,
    )
    print(score)


if __name__ == "__main__":
    test_compute_score()
