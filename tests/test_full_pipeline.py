import os
import json
from pprint import pprint
from typing import Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from evaluate._lp_utils import extract_python_code
import evaluate.evaluate_code as evaluate_code
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
    prompt = """Build a model for the following optimization problem and implement the model in Gurobi Optimizer(using `gurobipy` package). The requirement is as following
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

def test_full_pipeline(data_dir,sample_data,model_name,save_path):
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
    
            

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test with a specific model")
    parser.add_argument("--model_name", type=str, default = 'gpt-4o-mini', help="Name of the tested model")
    parser.add_argument("--batch_size", type=int, default = 8, help="batch size")
    parser.add_argument("--save_path", type=str, help = "path to save log files")
    
    args = parser.parse_args()
    batch_size = args.batch_size
    full_data = []
    # change to your own data path
    data_dir = "data/bench4opt_mix"
    # change the number of samples to test
    #test_full_pipeline(data_dir,get_sample(data_dir,12), args.model_name,args.save_path)

    for i in range(506):
        full_data.append(get_sample(data_dir,i))    

    all_results = []

    for i in tqdm(range(0, 506, batch_size)):
        batch_data = full_data[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            # map 返回一个迭代器，里面就是每个函数调用的返回值
            results = list(
                executor.map(
                    lambda p: test_full_pipeline(data_dir, p, args.model_name, args.save_path),
                    batch_data
                )
            )
            all_results.extend(results)  # 把 batch 的结果加到总结果里

    # 保存到 json
    with open(f"{args.save_path}", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)             
