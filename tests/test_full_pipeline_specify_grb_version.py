import os
import json
from pprint import pprint
from typing import Tuple, Dict, List, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from evaluate._lp_utils import extract_python_code
import evaluate.evaluate_code as evaluate_code
import time
import threading

def get_sample(data_dir: str, sample_id: int) -> dict:
    """
    Load a sample from the data path by its ID.
    
    Args:
        data_dir: Path to the data directory
        sample_id: ID of the sample to retrieve
        
    Returns:
        dict: Sample data dictionary
    """
    data = [json.loads(line) for line in open(os.path.join(data_dir, 'test.jsonl'), 'r')]
    sample_data = data[sample_id]

    # sample_data is a dictionary with the following key-value pairs:
    sample_data = {
        "id": sample_data["id"],
        "data_path": sample_data["data_path"],     # LP问题的数据路径
        "problem": sample_data["problem"],         # LP问题的描述
        "reference_lp_path": sample_data["reference_lp_path"], # LP问题的参考解
        "problem_type": sample_data["problem_type"],      # LP/MILP
        "problem_class": sample_data["problem_class"],   # 问题类型
    }

    return sample_data


def generate_prompt(sample_data: dict) -> str:
    """
    Generate a prompt based on the sample data.
    
    Args:
        sample_data: Dictionary containing sample information
        
    Returns:
        dict: Updated sample data with prompt added
    """
    prompt = """You are expert in optimization modeling and gurobipy. Build a model for the following optimization problem and implement the model in Gurobi Solver(using `gurobipy` package version 11.0.3). The requirement is as following
    1. Do not optimize or solve the model, instead save the model in `.lp` file.
    2. You should specify the upper and lower bounds for all variables.
    3. Be careful with the dtype and dimension of variables and parameters.
    4. Be careful not use any extra parameters that are not in the data file. If you have to use, make sure to define them clearly.
    5. Output the python code only. Do not output anything else. 
    6. Make sure your output code can be directly run in Gurobi Optimizer and save the desired `.lp` file"""
    
    problem = sample_data['problem']
    prompt = prompt + f"\n\n{problem}"

    return prompt

def generate_completion(sample_data: dict, model_name: str, **generation_kwargs) -> str:
    """
    Generate a completion using the specified model.
    
    Args:
        sample_data: Dictionary containing sample information and prompt
        model_name: Name of the model to use
        **generation_kwargs: Additional keyword arguments for generation
        
    Returns:
        dict: Updated sample data with completion added
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    kwargs = {
        "model": model_name,
        #"temperature": 0.0,
        #"top_p": 1.0,
        #"max_tokens": 16384,
        "seed": 42
    }
    for key, value in generation_kwargs.items():
        kwargs[key] = value
    
    request_output = client.chat.completions.create(
        messages=[{"role": "user", "content": sample_data['prompt']}],
        **kwargs
    )

    completion = request_output.choices[0].message.content
    return completion


def evaluate_completion(data_dir: str, sample_data: dict, model_name: str) -> Tuple[str, int, str]:
    """
    Evaluate the quality of the completion.
    
    Args:
        data_dir: Path to the data directory
        sample_data: Dictionary containing sample information and completion
        
    Returns:
        dict: Updated sample data with evaluation metrics
    """
    from evaluate._lp_utils import extract_python_code


    code = extract_python_code(sample_data["completion"])
    
    print("*"*10, "extracted code", "*"*10)
    #print(code)

    data_path = os.path.join(data_dir, sample_data["data_path"])
    reference_lp_path = os.path.join(data_dir, sample_data["reference_lp_path"])
    code_eval_result, wl_eval_result, equivalence_check_time = evaluate_code.evaluate_code_re(
                    model_name=model_name,
                    data_dir=data_dir,
                    code=code,
                    data_path=data_path,
                    reference_lp_path=reference_lp_path,
                    verbose=True,  # 改为False减少输出,
                    problem_id = sample_data['id'].split("_")[1]
                )

    if code_eval_result["success"]:
        code_success = True
        code_error_msg = ""
        code = code_eval_result["lp_file_path"]
        if wl_eval_result["success"]:
            wl_success = True
            wl_error_msg = wl_eval_result['message']
        else: 
            wl_success = False
            wl_error_msg = wl_eval_result['message']
    else:
        code_success = False
        code_error_msg = code_eval_result['message']
        code = code_eval_result["lp_file_path"]
        wl_success = False
        wl_error_msg = ""

    # Set reward for code generation
    code_reward = 1.0 if code_success else 0.0
    # Set reward for or modeling 
    wl_reward = 1.0 if wl_success else 0.0

    reward = {'code_reward':code_reward,"wl_reward":wl_reward}

    verification = {"code_verification":code_error_msg,
                    "wl_verification":wl_error_msg}
    # Update sample data with results
    return (
        code,
        reward,
        verification,
        equivalence_check_time
    )

def test_full_pipeline(data_dir, sample_data, model_name, save_path):
    """Test the full evaluation pipeline"""
    # Run the pipeline
    # 功能1: 生成提示
    prompt = generate_prompt(sample_data)
    print("*"*10,"Test generate_prompt success","*"*10)
    sample_data['prompt'] = prompt
    # 功能2: 生成完成
    sample_data['completion'] = generate_completion(sample_data, model_name)
    # 功能3: 评估完成
    code, reward, verification, equivalence_check_time = evaluate_completion(data_dir, sample_data, model_name)
    print("*"*10,"Test evaluate_completion success","*"*10)
    sample_data['code'] = code
    sample_data['reward'] = reward
    sample_data['verification'] = verification
    sample_data['equivalence_check_time'] = equivalence_check_time
    return sample_data


def load_existing_results(save_path: str) -> Dict[str, dict]:
    """
    加载现有的结果文件，返回已完成样本的字典
    
    Args:
        save_path: 结果文件路径
        
    Returns:
        Dict[str, dict]: 已完成样本的字典，key为sample_id
    """
    if not os.path.exists(save_path):
        print(f"结果文件 {save_path} 不存在，将从头开始")
        return {}
    
    try:
        with open(save_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 将结果转换为以sample_id为key的字典
        completed_samples = {}
        for result in results:
            if 'id' in result:
                completed_samples[result['id']] = result
        
        print(f"成功加载 {len(completed_samples)} 个已完成的结果")
        return completed_samples
    
    except Exception as e:
        print(f"加载结果文件时出错: {e}")
        return {}


def save_results_incremental(all_results: List[dict], save_path: str, lock: threading.Lock):
    """
    增量保存结果到文件
    
    Args:
        all_results: 所有结果列表
        save_path: 保存路径
        lock: 线程锁
    """
    with lock:
        try:
            # 创建备份
            if os.path.exists(save_path):
                backup_path = save_path + ".backup"
                os.rename(save_path, backup_path)
            
            # 保存新结果
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            # 删除备份
            if os.path.exists(save_path + ".backup"):
                os.remove(save_path + ".backup")
                
            print(f"结果已保存到 {save_path}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
            # 恢复备份
            if os.path.exists(save_path + ".backup"):
                os.rename(save_path + ".backup", save_path)


def get_completed_sample_ids(completed_samples: Dict[str, dict]) -> Set[str]:
    """
    获取已完成样本的ID集合
    
    Args:
        completed_samples: 已完成样本的字典
        
    Returns:
        Set[str]: 已完成样本ID的集合
    """
    return set(completed_samples.keys())


def filter_pending_samples(full_data: List[dict], completed_sample_ids: Set[str]) -> List[dict]:
    """
    过滤出待处理的样本
    
    Args:
        full_data: 所有样本数据
        completed_sample_ids: 已完成样本ID的集合
        
    Returns:
        List[dict]: 待处理的样本列表
    """
    pending_samples = []
    for sample in full_data:
        if sample['id'] not in completed_sample_ids:
            pending_samples.append(sample)
    
    print(f"总样本数: {len(full_data)}, 已完成: {len(completed_sample_ids)}, 待处理: {len(pending_samples)}")
    return pending_samples


def merge_results(existing_results: Dict[str, dict], new_results: List[dict]) -> List[dict]:
    """
    合并现有结果和新结果
    
    Args:
        existing_results: 现有结果字典
        new_results: 新结果列表
        
    Returns:
        List[dict]: 合并后的结果列表
    """
    # 将新结果添加到现有结果中
    for result in new_results:
        existing_results[result['id']] = result
    
    # 按原始顺序排序
    all_results = []
    for sample_id in sorted(existing_results.keys()):
        all_results.append(existing_results[sample_id])
    
    return all_results


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test with a specific model (with resume support)")
    parser.add_argument("--model_name", type=str, default='gpt-4o-mini', help="Name of the tested model")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--save_path", type=str, help="path to save log files")
    parser.add_argument("--save_interval", type=int, default=10, help="save results every N batches")
    parser.add_argument("--data_dir", type=str, default="/Users/zhuziwei/Downloads/ICLR_exp_副本/data/bench4opt_mix", help="path to data directory")
    
    args = parser.parse_args()
    batch_size = args.batch_size
    save_interval = args.save_interval
    
    # 检查保存路径
    if not args.save_path:
        print("错误: 必须指定 --save_path 参数")
        exit(1)
    
    # 创建保存目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 加载现有结果
    existing_results = load_existing_results(args.save_path)
    completed_sample_ids = get_completed_sample_ids(existing_results)
    
    # 准备所有样本数据
    full_data = []
    data_dir = args.data_dir
    
    for i in range(417):
        full_data.append(get_sample(data_dir, i))
    
    # 过滤出待处理的样本
    pending_samples = filter_pending_samples(full_data, completed_sample_ids)
    
    if not pending_samples:
        print("所有样本都已完成评测！")
        exit(0)
    
    # 初始化结果列表
    all_results = list(existing_results.values())
    new_results = []
    
    # 创建线程锁用于安全保存
    save_lock = threading.Lock()
    
    # 处理待处理的样本
    total_batches = (len(pending_samples) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(pending_samples), batch_size), desc="处理批次"):
        batch_data = pending_samples[batch_idx:batch_idx + batch_size]
        
        try:
            with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                # 处理当前批次
                results = list(
                    executor.map(
                        lambda p: test_full_pipeline(data_dir, p, args.model_name, args.save_path),
                        batch_data
                    )
                )
                new_results.extend(results)
                all_results.extend(results)
                
                print(f"批次 {batch_idx//batch_size + 1}/{total_batches} 完成，处理了 {len(results)} 个样本")
                
        except Exception as e:
            print(f"处理批次时出错: {e}")
            # 即使出错也要保存已处理的结果
            pass
        
        # 定期保存结果
        if (batch_idx // batch_size + 1) % save_interval == 0:
            save_results_incremental(all_results, args.save_path, save_lock)
            print(f"已保存中间结果，完成 {len(all_results)} 个样本")
    
    # 最终保存
    save_results_incremental(all_results, args.save_path, save_lock)
    print(f"评测完成！总共处理了 {len(all_results)} 个样本，结果已保存到 {args.save_path}")
