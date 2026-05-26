import os
import sys
import json
from pprint import pprint
from typing import Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate._lp_utils import extract_python_code
import evaluate.evaluate_code as evaluate_code

def load_resume_data(resume_file_path: str) -> list:
    """
    从resume.json文件加载数据
    
    Args:
        resume_file_path: resume.json文件路径
        
    Returns:
        list: 包含所有样本数据的列表
    """
    with open(resume_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def evaluate_completion_with_solver(data_dir: str, sample_data: dict, model_name: str) -> Tuple[str, dict, dict, dict]:
    """
    使用solver evaluation评估completion的质量
    
    Args:
        data_dir: 数据目录路径
        sample_data: 包含样本信息和completion的字典
        model_name: 模型名称
        
    Returns:
        tuple: (code, reward, verification, solver_check_time)
    """
    from evaluate._lp_utils import extract_python_code

    code = extract_python_code(sample_data["completion"])
    
    print("*"*10, "extracted code", "*"*10)
    #print(code)

    data_path = os.path.join(data_dir, sample_data["data_path"])
    reference_lp_path = os.path.join(data_dir, sample_data["reference_lp_path"])
    
    # 使用solver evaluation而不是WL test
    code_eval_result, solver_eval_result, solver_check_time = evaluate_code.evaluate_code_solver(
                    model_name=model_name,
                    data_dir=data_dir,
                    code=code,
                    data_path=data_path,
                    reference_lp_path=reference_lp_path,
                    verbose=True,  # 改为False减少输出
                    problem_id = sample_data['id'].split("_")[1]
                )

    if code_eval_result["success"]:
        code_success = True
        code_error_msg = ""
        code = code_eval_result["lp_file_path"]
        if solver_eval_result["success"]:
            solver_success = True
            solver_error_msg = solver_eval_result['message']
        else: 
            solver_success = False
            solver_error_msg = solver_eval_result['message']
    else:
        code_success = False
        code_error_msg = code_eval_result['message']
        code = code_eval_result["lp_file_path"]
        solver_success = False
        solver_error_msg = ""

    # Set reward for code generation
    code_reward = 1.0 if code_success else 0.0
    # Set reward for solver evaluation 
    solver_reward = 1.0 if solver_success else 0.0

    reward = {'code_reward': code_reward, "solver_reward": solver_reward}

    verification = {"code_verification": code_error_msg,
                    "solver_verification": solver_error_msg}
    
    return (
        code,
        reward,
        verification,
        solver_check_time
    )

def test_solver_pipeline(data_dir, sample_data, model_name, save_path):
    """测试solver evaluation pipeline"""
    # 功能: 评估completion
    code, reward, verification, solver_check_time = evaluate_completion_with_solver(data_dir, sample_data, model_name)
    print("*"*10,"Test solver evaluation success","*"*10)
    
    # 更新sample_data
    sample_data['code'] = code
    sample_data['reward'] = reward
    sample_data['verification'] = verification
    sample_data['solver_check_time'] = solver_check_time
    
    return sample_data

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run solver evaluation with a specific model")
    parser.add_argument("--model_name", type=str, default = 'deepseek-v3', help="Name of the tested model")
    parser.add_argument("--batch_size", type=int, default = 8, help="batch size")
    parser.add_argument("--resume_file", type=str, help = "path to resume json file")
    parser.add_argument("--save_path", type=str, help = "path to save log files")
    
    args = parser.parse_args()
    batch_size = args.batch_size
    
    # 从resume文件加载数据
    resume_data = load_resume_data(args.resume_file)
    print(f"Loaded {len(resume_data)} samples from {args.resume_file}")
    
    # change to your own data path
    data_dir = "/Users/zhuziwei/Downloads/ICLR_exp_副本/data/bench4opt_mix_filtered"

    all_results = []

    for i in tqdm(range(0, len(resume_data), batch_size)):
        batch_data = resume_data[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            # map 返回一个迭代器，里面就是每个函数调用的返回值
            results = list(
                executor.map(
                    lambda p: test_solver_pipeline(data_dir, p, args.model_name, args.save_path),
                    batch_data
                )
            )
            all_results.extend(results)  # 把 batch 的结果加到总结果里

    # 保存到 json
    with open(f"{args.save_path}", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"Solver evaluation results saved to {args.save_path}")
