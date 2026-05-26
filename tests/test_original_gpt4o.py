import os
import json
from pprint import pprint
from typing import Tuple
import time
import traceback
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor


def get_sample(data_dir: str, sample_id: int, file_name: str = "test.jsonl") -> dict:
    """
    Load a sample from the data path by its ID.

    Args:
        data_dir: Path to the data directory
        sample_id: ID of the sample to retrieve
        file_name: The name of the jsonl file to read from

    Returns:
        dict: Sample data dictionary
    """
    # 打印数据文件的绝对路径
    file_path = os.path.abspath(os.path.join(data_dir, file_name))
    
    data = [
        json.loads(line) for line in open(os.path.join(data_dir, file_name), "r")
    ]
    sample_data = data[sample_id]

    # sample_data is a dictionary with the following key-value pairs:
    sample_data = {
        "id": sample_data["id"],
        "data_path": sample_data["data_path"],  # LP问题的数据路径
        "problem": sample_data["problem"],  # LP问题的描述
        "reference_lp_path": sample_data["reference_lp_path"],  # LP问题的参考解
        "problem_type": sample_data["problem_type"],  # LP/MILP
        "problem_class": sample_data["problem_class"],  # 问题类型
        "wp_type":sample_data["wp_type"]
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
    prompt = "Build a model for the following opimization problem and implement the model in Gurobi Optimizer. Output the python code only. Do not output anything else. Make sure your output code can be directly run in Gurobi Optimizer and solve for the final solution."

    problem = sample_data["problem"]
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
    print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )

    kwargs = {
        "model": model_name,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "seed": 42,
    }
    for key, value in generation_kwargs.items():
        kwargs[key] = value

    request_output = client.chat.completions.create(
        messages=[{"role": "user", "content": sample_data["prompt"]}], **kwargs
    )

    completion = request_output.choices[0].message.content
    return completion


def evaluate_completion(
    data_dir: str, sample_data: dict, verbose: bool = False
) -> Tuple[str, int, str]:
    """
    Evaluate the quality of the completion.

    Args:
        data_dir: Path to the data directory
        sample_data: Dictionary containing sample information and completion

    Returns:
        dict: Updated sample data with evaluation metrics
    """
    from evaluate._lp_utils import extract_python_code,ensure_imports
    from evaluate.evaluate_code import evaluate_code_v2
    response_path = sample_data["reference_lp_path"].split('/')[1].replace('.lp','.txt').replace('model','gpt4o')
    response_path = f'txt_gpt4o/{response_path}'
    response_path = os.path.join(data_dir,response_path)
    with open(response_path, 'r') as f:
        response = f.read()
    code = extract_python_code(response)
    print("*" * 10, "extracted code", "*" * 10)
    code = ensure_imports(code)
    print(code)

    problem_id = sample_data['id']
    data_path = os.path.join(data_dir, sample_data["data_path"])
    reference_lp_path = os.path.join(data_dir, sample_data["reference_lp_path"])

    code_eval_result, wl_eval_result = evaluate_code_v2(
        code, data_path, reference_lp_path, verbose=verbose, problem_id=problem_id
    )

    if code_eval_result["success"]:
        code_success = True
        code_error_msg = ""
        code = code_eval_result["lp_file_path"]
        if wl_eval_result["success"]:
            wl_success = True
            wl_error_msg = wl_eval_result["message"]
        else:
            wl_success = False
            wl_error_msg = wl_eval_result["message"]
    else:
        code_success = False
        code_error_msg = code_eval_result["message"]
        code = code_eval_result["lp_file_path"]
        wl_success = False
        wl_error_msg = ""

    # Set reward for code generation
    code_reward = 1.0 if code_success else 0.0
    # Set reward for or modeling
    wl_reward = 1.0 if wl_success else 0.0

    reward = {"code_reward": code_reward, "wl_reward": wl_reward}

    verification = {
        "code_verification": code_error_msg,
        "wl_verification": wl_error_msg
    }
    # Update sample data with results
    return (
        code,
        reward,
        verification,
    )


def get_total_samples(data_dir: str, file_name: str = "test.jsonl") -> int:
    """获取数据文件中的总样本数
    
    Args:
        data_dir: 数据目录路径
        file_name: jsonl文件名
    """
    with open(os.path.join(data_dir, file_name), "r") as f:
        return sum(1 for _ in f)


def process_single_sample(args) -> dict:
    """处理单个样本的函数
    
    Args:
        args: 包含处理单个样本所需的所有参数的元组
            (data_dir, sample_id, model_name, file_name, verbose)
    
    Returns:
        dict: 处理结果
    """
    data_dir, curr_sample_id, model_name, file_name, verbose = args
    print(f"\n{'='*20} 测试样本 {curr_sample_id} {'='*20}")
    step_times = {}
    total_start_time = time.time()

    try:
        # Run the pipeline
        # 功能0: 获取样本
        start_time = time.time()
        sample_data = get_sample(data_dir, curr_sample_id, file_name)
        step_times['get sample'] = time.time() - start_time
        # print("Test get_sample success")
        # print("*" * 10, "sample_data", "*" * 10)
        # pprint(sample_data)

        # 功能1: 生成提示
        start_time = time.time()
        prompt = generate_prompt(sample_data)
        sample_data["prompt"] = prompt
        step_times['prompt generation'] = time.time() - start_time
        print("Test generate_prompt success")
        # print("*" * 10, "prompt", "*" * 10)
        # print(prompt)

        # 功能2: 生成完成
        #start_time = time.time()
        #completion = generate_completion(sample_data, model_name)
        #sample_data["completion"] = completion
        #step_times['finish generation'] = time.time() - start_time
        #print("Test generate_completion success")
        # print("*" * 10, "completion", "*" * 10)
        # print(completion)

        # 功能3: 评估完成
        start_time = time.time()
        code, reward, verification = evaluate_completion(
            data_dir, sample_data, verbose=verbose
        )
        sample_data["code"] = code
        sample_data["reward"] = reward
        sample_data["verification"] = verification
        step_times['finish evaluation'] = time.time() - start_time
        print("Test evaluate_completion success")
        # print("*" * 10, "code", "*" * 10)
        # print(code)
        # print("*" * 10, "reward", "*" * 10)
        print(reward)
        print("*" * 10, "verification", "*" * 10)
        print(verification)

        # Verify the result contains all expected fields
        assert "id" in sample_data
        assert "prompt" in sample_data
        assert "code" in sample_data
        assert "reward" in sample_data
        assert "verification" in sample_data

        # 打印时间统计
        total_time = time.time() - total_start_time
        print("\n" + "="*50)
        print("时间统计:")
        print("-"*30)
        for step, duration in step_times.items():
            print(f"{step:<10}: {duration:.2f}秒")
        print("-"*30)
        print(f"总耗时: {total_time:.2f}秒")
        print("="*50)
        result = {
            "model_name": model_name,
            "sample_id": curr_sample_id,
            "reward": reward,
            "time": total_time,
            "step_times": step_times,
            "verification": verification,
            "success": True,
            "problem_type": sample_data["problem_type"],  # LP/MILP
            "problem_class": sample_data["problem_class"],  # 问题类型
            "wp_type": sample_data["wp_type"]
        }
        
        # 返回结果
    except Exception as e:
        print(f"\n{'='*20} 样本 {curr_sample_id} 处理失败 {'='*20}")
        print("错误信息:", str(e))
        print("\n完整堆栈跟踪:")
        print(traceback.format_exc())
        result = {
            "model_name": model_name,
            "sample_id": curr_sample_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }
    model_name_ =model_name.replace("/","-")
    save_path = f"outputs/test_{model_name_}_bench_ori.json"
    with open(f"{save_path}", "a") as f:
        json.dump(result, f, indent=4)
    return result


def main(data_dir: str, sample_id: int = None, model_name: str = "gpt-4o-mini", file_name: str = "test.jsonl", verbose: bool = False):
    """Test the full evaluation pipeline
    
    Args:
        data_dir: 数据目录路径
        sample_id: 指定样本ID，如果为None则测试所有样本
        model_name: 模型名称
        file_name: 要测试的jsonl文件名
        verbose: 是否显示详细信息
    """
    if sample_id is not None:
        # 测试单个样本
        sample_ids = [sample_id]
    else:
        # 测试所有样本
        total_samples = get_total_samples(data_dir, file_name)
        sample_ids = range(total_samples)
        print(f"将测试所有样本，共 {total_samples} 个")

    # 准备并行处理的参数
    process_args = [(data_dir, sid, model_name, file_name, verbose) for sid in sample_ids]
    
    # 使用进程池并行处理
    num_processes = 16#max_workers=os.cpu_count() * 2#min(cpu_count(), len(sample_ids))  # 使用CPU核心数和样本数中的较小值
    print(f"\n使用 {num_processes} 个线程并行处理...")

    with ThreadPoolExecutor(num_processes) as executor:
        all_results = executor.map(
            lambda p: process_single_sample(p), process_args
        )

    #with Pool(num_processes) as pool:
    #    all_results = pool.map(process_single_sample, process_args)

    # 打印总体统计
    if len(sample_ids) > 1:
        print("\n" + "="*50)
        print("总体统计:")
        print("-"*30)
        successful_samples = [r for r in all_results if r["success"]]
        failed_samples = [r for r in all_results if not r["success"]]
        print(f"总样本数: {len(sample_ids)}")
        print(f"成功样本数: {len(successful_samples)}")
        print(f"失败样本数: {len(failed_samples)}")
        
        if failed_samples:
            print("\n失败样本列表:")
            for r in failed_samples:
                print(f"样本 {r['sample_id']} - 错误: {r['error']}")
        
        if successful_samples:
            avg_time = sum(r["time"] for r in successful_samples) / len(successful_samples)
            print(f"\n平均耗时: {avg_time:.2f}秒")
            
            # 计算每个步骤的平均时间
            all_steps = set()
            for r in successful_samples:
                if "step_times" in r:
                    all_steps.update(r["step_times"].keys())
            
            if all_steps:
                print("\n各步骤平均耗时:")
                for step in sorted(all_steps):
                    step_times = [r["step_times"][step] for r in successful_samples if "step_times" in r and step in r["step_times"]]
                    if step_times:
                        avg_step_time = sum(step_times) / len(step_times)
                        print(f"{step:<10}: {avg_step_time:.2f}秒")
            
            avg_code_reward = sum(r["reward"]["code_reward"] for r in successful_samples) / len(successful_samples)
            avg_wl_reward = sum(r["reward"]["wl_reward"] for r in successful_samples) / len(successful_samples)
            print(f"\n平均代码生成奖励: {avg_code_reward:.2f}")
            print(f"平均建模奖励: {avg_wl_reward:.2f}")
        print("="*50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/liziniu/project/or_modeling/data/bench4opt",
    )
    parser.add_argument("--sample_id", type=int, default=None,
                      help="指定样本ID，不指定则测试所有样本")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--file_name", type=str, default="test.jsonl",
                      help="要测试的jsonl文件名")

    args = parser.parse_args()

    main(args.data_dir, args.sample_id, args.model_name, args.file_name, verbose=True)
