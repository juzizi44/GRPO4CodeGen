import json
import os

# 文件路径
file_paths = [
    "/data/ytan089/GRPO4CodeGen_v2/dataset/score/gpt-4.1-mini-2025-04-14-Instruct_score_result.jsonl",
    "/data/ytan089/GRPO4CodeGen_v2/dataset/score/Qwen2.5-Coder-7B-Instruct_score_result.jsonl",
    # "/data/ytan089/GRPO4CodeGen_v2/dataset/score/grpo_use_weight_net_step1776_score_result.jsonl",
    # "/data/ytan089/GRPO4CodeGen_v2/dataset/score/grpo_fix_weight_step_score_result.jsonl",
    # "/data/ytan089/GRPO4CodeGen_v2/dataset/score/grpo_use_weight_net_step2516_score_result.jsonl",
    # "/data/ytan089/GRPO4CodeGen_v2/dataset/score/20250529_all_weightnet_sorting_1e-5_1058_score_result.jsonl",
    # "/data/ytan089/GRPO4CodeGen_v2/dataset/score/20250603_all_weightnet_5772_score_result.jsonl",
    "/data/ytan089/GRPO4CodeGen_v2/dataset/score/grpo_20250702_all_data_result.jsonl"
]

# 文件名缩写，用于结果展示，与上面是一一对应的。
file_labels = [
    "GPT-4.1-mini",
    # "GRPO-Qwen-1344",
    "Qwen-7B",
    # "GRPO-Qwen-use-weight-net",
    # "GRPO-Qwen-fix-weight",
    "GRPO",
]

# 指标列表
metrics = ["pass_rate", "avg_time_consumed", "avg_tracemalloc_peak", "avg_cpu_instruction_count",
          "code_comment_score", "llm_comment_score", "comment_score", "maintainability_score"]

def load_data(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    return data

def calculate_averages(data):
    """计算平均指标"""
    total_items = len(data)
    sums = {metric: 0 for metric in metrics}
    valid_counts = {metric: 0 for metric in metrics}

    for item in data:
        for metric in metrics:
            if metric in item:
                # pass_rate和评分类指标：不管是不是0都加
                if metric in ["pass_rate", "code_comment_score", "llm_comment_score", "comment_score", "maintainability_score"]:
                    sums[metric] += item[metric]
                    valid_counts[metric] += 1
                # 其他指标：仅统计非零值
                elif item[metric] != 0:
                    sums[metric] += item[metric]
                    valid_counts[metric] += 1

    averages = {}
    for metric in metrics:
        if metric in ["pass_rate", "code_comment_score", "llm_comment_score", "comment_score", "maintainability_score"]:
            averages[metric] = sums[metric] / total_items if total_items > 0 else 0
        else:
            averages[metric] = sums[metric] / valid_counts[metric] if valid_counts[metric] > 0 else 0

    return averages

def compare_metrics_by_question(all_data):
    """比较每个问题的指标，统计各文件的最高值次数"""
    best_counts = {label: {metric: 0 for metric in metrics} for label in file_labels}
    
    # 按问题ID组织数据
    questions = {}
    for i, data in enumerate(all_data):
        for item in data:
            q_id = item["question_id"]
            if q_id not in questions:
                questions[q_id] = [{} for _ in range(len(all_data))]
            questions[q_id][i] = item
    
    # 对每个问题比较指标
    for q_id, results in questions.items():
        for metric in metrics:
            best_value = -1
            best_indices = []

            for i, result in enumerate(results):
                if metric not in result:
                    continue

                value = result[metric]

                # 忽略值为 0 的情况（仅对非评分类指标）
                if metric not in ["pass_rate", "code_comment_score", "llm_comment_score", "comment_score", "maintainability_score"] and value == 0:
                    continue

                # 对于耗时/内存等，值越小越好，转为 1/value
                if metric in ["avg_time_consumed", "avg_tracemalloc_peak", "avg_cpu_instruction_count"]:
                    value = 1 / value if value > 0 else 0

                if value > best_value:
                    best_value = value
                    best_indices = [i]
                elif value == best_value:
                    best_indices.append(i)

            # 平分权重
            for idx in best_indices:
                if idx < len(file_labels):
                    best_counts[file_labels[idx]][metric] += 1 / len(best_indices)

    return best_counts

# 主程序
def main():
    all_data = []
    all_averages = []
    
    print("加载数据文件...")
    for path in file_paths:
        if os.path.exists(path):
            data = load_data(path)
            all_data.append(data)
            averages = calculate_averages(data)
            all_averages.append(averages)
        else:
            print(f"文件不存在: {path}")
            return
    
    print("\n每个模型的平均指标:")
    print("-" * 160)
    print(f"{'模型':<15} | {'通过率':<10} | {'平均耗时(秒)':<12} | {'内存峰值':<12} | {'CPU指令数':<15} | {'代码注释':<10} | {'LLM注释':<10} | {'注释总分':<10} | {'可维护性':<10}")
    print("-" * 160)
    
    for i, label in enumerate(file_labels):
        avg = all_averages[i]
        print(f"{label:<15} | {avg['pass_rate']:<10.4f} | {avg['avg_time_consumed']:<12.6f} | "
              f"{avg['avg_tracemalloc_peak']:<12.2f} | {avg['avg_cpu_instruction_count']:<15.2f} | "
              f"{avg['code_comment_score']:<10.4f} | {avg['llm_comment_score']:<10.4f} | "
              f"{avg['comment_score']:<10.4f} | {avg['maintainability_score']:<10.4f}")
    
    print("\n\n最佳指标统计(每个指标获得最高值的次数):")
    print("-" * 160)
    best_counts = compare_metrics_by_question(all_data)
    
    print(f"{'模型':<15} | {'通过率':<10} | {'耗时最少':<12} | {'内存最少':<12} | {'CPU最少':<15} | "
          f"{'代码注释':<10} | {'LLM注释':<10} | {'注释总分':<10} | {'可维护性':<10}")
    print("-" * 160)
    
    for label in file_labels:
        counts = best_counts[label]
        print(f"{label:<15} | {counts['pass_rate']:<10.1f} | {counts['avg_time_consumed']:<12.1f} | "
              f"{counts['avg_tracemalloc_peak']:<12.1f} | {counts['avg_cpu_instruction_count']:<15.1f} | "
              f"{counts['code_comment_score']:<10.1f} | {counts['llm_comment_score']:<10.1f} | "
              f"{counts['comment_score']:<10.1f} | {counts['maintainability_score']:<10.1f}")

if __name__ == "__main__":
    main()