#!/bin/bash

# =================================================================================
# Shell 脚本: 全面基准测试 CRC 模型
# ---------------------------------------------------------------------------------
# 该脚本将在所有4个ETT数据集上，分别使用 Informer, Autoformer, 和 TimesNet
# 作为基线模型，来测试 CRC 校正器的性能。总共会运行 12 个实验。
# =================================================================================

echo "Starting Comprehensive Benchmark for CRC Model..."

# --- 定义通用参数 ---
# 预测范围等参数保持一致，方便对比
SEQ_LEN=96
PRED_LEN=24


# ==================================================
#            数据集: ETTh1 (小时级)
# ==================================================
DATA_NAME="ETTh1"
DATA_FILE="ETTh1.csv"
ROOT_PATH="./dataset/ETT-small/"
FREQ="h"

echo -e "\n\n===== Running on Dataset: $DATA_NAME ====="

# --- 基线: Informer ---
echo -e "\n--- Baseline: Informer on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_Informer --model CRC --baseline_model Informer --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'CRC with Informer on ETTh1' --itr 1

# --- 基线: Autoformer ---
echo -e "\n--- Baseline: Autoformer on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_Autoformer --model CRC --baseline_model Autoformer --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'CRC with Autoformer on ETTh1' --itr 1

# --- 基线: TimesNet ---
echo -e "\n--- Baseline: TimesNet on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_TimesNet --model CRC --baseline_model TimesNet --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --d_ff 32 --des 'CRC with TimesNet on ETTh1' --itr 1


# ==================================================
#            数据集: ETTh2 (小时级)
# ==================================================
DATA_NAME="ETTh2"
DATA_FILE="ETTh2.csv"
# ROOT_PATH 和 FREQ 不变

echo -e "\n\n===== Running on Dataset: $DATA_NAME ====="

# --- 基线: Informer ---
echo -e "\n--- Baseline: Informer on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_Informer --model CRC --baseline_model Informer --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'CRC with Informer on ETTh2' --itr 1

# --- 基线: Autoformer ---
echo -e "\n--- Baseline: Autoformer on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_Autoformer --model CRC --baseline_model Autoformer --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'CRC with Autoformer on ETTh2' --itr 1

# --- 基线: TimesNet ---
echo -e "\n--- Baseline: TimesNet on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_TimesNet --model CRC --baseline_model TimesNet --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --d_ff 32 --des 'CRC with TimesNet on ETTh2' --itr 1


# ==================================================
#            数据集: ETTm1 (15分钟级)
# ==================================================
DATA_NAME="ETTm1"
DATA_FILE="ETTm1.csv"
FREQ="t" # <-- 频率改为分钟级

echo -e "\n\n===== Running on Dataset: $DATA_NAME ====="

# --- 基线: Informer ---
echo -e "\n--- Baseline: Informer on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_Informer --model CRC --baseline_model Informer --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'CRC with Informer on ETTm1' --itr 1

# --- 基线: Autoformer ---
echo -e "\n--- Baseline: Autoformer on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_Autoformer --model CRC --baseline_model Autoformer --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'CRC with Autoformer on ETTm1' --itr 1

# --- 基线: TimesNet ---
echo -e "\n--- Baseline: TimesNet on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_TimesNet --model CRC --baseline_model TimesNet --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --d_ff 32 --des 'CRC with TimesNet on ETTm1' --itr 1


# ==================================================
#            数据集: ETTm2 (15分钟级)
# ==================================================
DATA_NAME="ETTm2"
DATA_FILE="ETTm2.csv"
# FREQ 仍为 "t"

echo -e "\n\n===== Running on Dataset: $DATA_NAME ====="

# --- 基线: Informer ---
echo -e "\n--- Baseline: Informer on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_Informer --model CRC --baseline_model Informer --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'CRC with Informer on ETTm2' --itr 1

# --- 基线: Autoformer ---
echo -e "\n--- Baseline: Autoformer on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_Autoformer --model CRC --baseline_model Autoformer --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'CRC with Autoformer on ETTm2' --itr 1

# --- 基线: TimesNet ---
echo -e "\n--- Baseline: TimesNet on $DATA_NAME ---"
python -u run.py --task_name short_term_forecast --is_training 1 --root_path $ROOT_PATH --data_path $DATA_FILE --model_id ${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_CRC_TimesNet --model CRC --baseline_model TimesNet --data $DATA_NAME --features M --freq $FREQ --seq_len $SEQ_LEN --pred_len $PRED_LEN --label_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --d_ff 32 --des 'CRC with TimesNet on ETTm2' --itr 1


echo "=================================================="
echo "All 12 benchmark experiments are complete!"
echo "=================================================="