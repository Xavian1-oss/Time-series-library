import re
import os
import pandas as pd

def extract_metrics_from_log(file_path):
    """
    (此函数与上一版相同)
    从指定的日志文件中提取基线模型和CRC纠偏后的MSE与MAE。
    """
    metrics = {
        'baseline_mse': None,
        'baseline_mae': None,
        'corrected_mse': None,
        'corrected_mae': None
    }
    baseline_pattern = re.compile(r"基线模型.*MSE: (\d+\.\d+).*MAE: (\d+\.\d+)")
    corrected_pattern = re.compile(r"CRC 纠偏后.*MSE: (\d+\.\d+).*MAE: (\d+\.\d+)")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                baseline_match = baseline_pattern.search(line)
                if baseline_match:
                    metrics['baseline_mse'] = float(baseline_match.group(1))
                    metrics['baseline_mae'] = float(baseline_match.group(2))
                
                corrected_match = corrected_pattern.search(line)
                if corrected_match:
                    metrics['corrected_mse'] = float(corrected_match.group(1))
                    metrics['corrected_mae'] = float(corrected_match.group(2))
        return metrics
    except Exception as e:
        print(f"读取文件 '{file_path}' 时出错: {e}")
        return None

def parse_filename(filename):
    """
    从文件名中解析出 数据集, 模型, 和 预测步长 (Horizon)。
    文件名格式假定为: ETTh1_crc_autoformer_m_pl192_dm64.log
    """
    try:
        # 移除 .log 后缀并按 '_' 分割
        parts = filename.replace('.log', '').split('_')
        
        dataset = parts[0]
        model = parts[2]
        
        # 寻找 'pl' 开头的部分来确定 horizon
        horizon = None
        for part in parts:
            if part.startswith('pl'):
                # 移除 'pl' 并转换为整数
                horizon = int(part[2:])
                break
        
        return dataset, model, horizon
    except Exception as e:
        print(f"  - 警告: 无法解析文件名 '{filename}'。跳过。错误: {e}")
        return None, None, None


def process_log_directory(root_directory):
    """
    处理指定目录下的所有 .log 文件，解析文件名和文件内容，并汇总结果。
    """
    if not os.path.isdir(root_directory):
        print(f"错误：目录 '{root_directory}' 不存在。")
        return

    all_results = []

    for filename in os.listdir(root_directory):
        if filename.endswith(".log"):
            print(f"正在处理文件: {filename}...")
            
            # 1. 从文件名解析元数据
            dataset, model, horizon = parse_filename(filename)
            if not dataset: # 如果解析失败，则跳过此文件
                continue

            # 2. 从文件内容提取性能指标
            full_path = os.path.join(root_directory, filename)
            metrics = extract_metrics_from_log(full_path)
            
            if metrics:
                # 3. 将所有信息合并到一个字典中
                result_row = {
                    'dataset': dataset,
                    'model': model,
                    'horizon': horizon,
                    'baseline_mse': metrics['baseline_mse'],
                    'baseline_mae': metrics['baseline_mae'],
                    'corrected_mse': metrics['corrected_mse'],
                    'corrected_mae': metrics['corrected_mae'],
                    'filename': filename # 保留完整文件名以供参考
                }
                all_results.append(result_row)

    if all_results:
        df = pd.DataFrame(all_results)
        
        # 调整最终输出的列顺序，使其更符合逻辑
        cols_order = [
            'dataset', 'model', 'horizon', 
            'baseline_mse', 'baseline_mae', 
            'corrected_mse', 'corrected_mae', 'filename'
        ]
        # 过滤掉不存在的列以增加稳健性
        existing_cols = [col for col in cols_order if col in df.columns]
        df = df[existing_cols]
        
        print("\n" + "="*30 + " 结 果 汇 总 " + "="*30)
        print(df)
        
        output_csv_path = os.path.join(root_directory, 'results_summary_detailed.csv')
        try:
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n结果已成功保存到: {output_csv_path}")
        except Exception as e:
            print(f"\n保存CSV文件时出错: {e}")
            
    else:
        print("\n在指定目录中没有找到任何有效的日志文件或性能指标。")

# --- 使用示例 ---

# !! 重要：请将这里的路径替换成你自己的文件夹路径 !!
log_directory_path = 'logs/multi_20250927_135258'

# 运行主处理函数
process_log_directory(log_directory_path)