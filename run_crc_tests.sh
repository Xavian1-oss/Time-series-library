#!/usr/bin/env bash
set -euo pipefail

# 选 GPU（如果是 MPS 或 CPU，可删掉这行）
export CUDA_VISIBLE_DEVICES=0

# 通用参数（只跑 M 模式；d_model=64）
DATASET=ETTh1
ROOT=./dataset/ETT-small/
CSV=ETTh1.csv

SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96

# M 模式下通道数
ENC_IN=7
DEC_IN=7
C_OUT=7

# 训练设置
EPOCHS=50
PATIENCE=7
BATCH=32
LR=1e-3

# CRC 相关
Q_VAL=3
K_VAL=24
TOP_K=5

# 模型规模（按你要求）
DMODEL=64
NHEADS=8
ELAYERS=2
DLAYERS=1
DFF=2048

# 是否计算 DTW / 反归一化（按需启用或注释）
USE_DTW=--use_dtw
INVERSE=--inverse

echo "================ 基线：TimesNet (CRC, M) ================"
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model CRC \
  --baseline_model TimesNet \
  --model_id etth1_crc_timesnet_m_pl${PRED_LEN}_dm${DMODEL} \
  --data ${DATASET} \
  --root_path ${ROOT} \
  --data_path ${CSV} \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in ${ENC_IN} \
  --dec_in ${DEC_IN} \
  --c_out ${C_OUT} \
  --d_model ${DMODEL} \
  --n_heads ${NHEADS} \
  --e_layers ${ELAYERS} \
  --d_layers ${DLAYERS} \
  --d_ff ${DFF} \
  --batch_size ${BATCH} \
  --learning_rate ${LR} \
  --train_epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  ${USE_DTW} ${INVERSE} \
  --q_val ${Q_VAL} \
  --k_val ${K_VAL} \
  --top_k ${TOP_K} \
  --des "CRC_TimesNet_M_d${DMODEL}"

echo "================ 基线：TimeMixer (CRC, M) ================"
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model CRC \
  --baseline_model TimeMixer \
  --model_id etth1_crc_timemixer_m_pl${PRED_LEN}_dm${DMODEL} \
  --data ${DATASET} \
  --root_path ${ROOT} \
  --data_path ${CSV} \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in ${ENC_IN} \
  --dec_in ${DEC_IN} \
  --c_out ${C_OUT} \
  --d_model ${DMODEL} \
  --n_heads ${NHEADS} \
  --e_layers ${ELAYERS} \
  --d_layers ${DLAYERS} \
  --d_ff ${DFF} \
  --batch_size ${BATCH} \
  --learning_rate ${LR} \
  --train_epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  ${USE_DTW} ${INVERSE} \
  --q_val ${Q_VAL} \
  --k_val ${K_VAL} \
  --top_k ${TOP_K} \
  --des "CRC_TimeMixer_M_d${DMODEL}"

echo "================ 基线：DLinear (CRC, M) ================"
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model CRC \
  --baseline_model DLinear \
  --model_id etth1_crc_dlinear_m_pl${PRED_LEN}_dm${DMODEL} \
  --data ${DATASET} \
  --root_path ${ROOT} \
  --data_path ${CSV} \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in ${ENC_IN} \
  --dec_in ${DEC_IN} \
  --c_out ${C_OUT} \
  --d_model ${DMODEL} \
  --n_heads ${NHEADS} \
  --e_layers ${ELAYERS} \
  --d_layers ${DLAYERS} \
  --d_ff ${DFF} \
  --batch_size ${BATCH} \
  --learning_rate ${LR} \
  --train_epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  ${USE_DTW} ${INVERSE} \
  --q_val ${Q_VAL} \
  --k_val ${K_VAL} \
  --top_k ${TOP_K} \
  --des "CRC_DLinear_M_d${DMODEL}"

echo "所有 CRC(M) 运行完成。"
