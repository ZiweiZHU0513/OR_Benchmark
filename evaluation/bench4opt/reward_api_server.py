#!/usr/bin/env python3
import os
import json
import argparse
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from logging.handlers import RotatingFileHandler

# 配置日志
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            "./logs/bench4opt_reward_api_server.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=3,  # 保留3个备份文件
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# 导入评估函数
from evaluation.bench4opt import reward_function

app = Flask(__name__)
CORS(app)  # 启用跨域请求支持


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "ok"})


@app.route("/compute_score", methods=["POST"])
def compute_score():
    """评估端点，接收代码并返回评估结果"""
    start_time = time.time()
    # try:
    # 获取请求数据
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # 提取必要的字段
    solution_str = data.get("solution_str")
    ground_truth = data.get("ground_truth")
    return_dict = data.get("return_dict", False)
    ensure_imports = data.get("ensure_imports", False)
    save_result = data.get("save_result", False)
    save_path = data.get("save_path", None)
    model_name = data.get("model_name", None)
    verbose = data.get("verbose", False)

    # 验证必要的字段
    if not solution_str:
        return jsonify({"error": "No solution_str provided"}), 400
    if not ground_truth:
        return jsonify({"error": "No ground_truth provided"}), 400

    ts = time.time()
    score = reward_function.compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        return_dict=return_dict,
        ensure_imports=ensure_imports,
        save_result=save_result,
        save_path=save_path,
        model_name=model_name,
        verbose=verbose,
    )
    elapsed_time = time.time() - ts

    # 记录结果和运行时间
    logger.info(f"Evaluation completed in {elapsed_time:.1f}s with reward={score}")

    # 返回结果
    return jsonify({"reward": score, "details": {"execution_time": elapsed_time}})


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Bench4Opt Reward Function Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Starting Bench4Opt Reward Function Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
