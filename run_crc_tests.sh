#!/usr/bin/env bash
set -euo pipefail

# ============== 基础设置 ==============
export CUDA_VISIBLE_DEVICES=0

SEQ_LEN=96
LABEL_LEN=48

ENC_IN=7
DEC_IN=7
C_OUT=7

EPOCHS=50
PATIENCE=7
BATCH=32
LR=1e-3

Q_VAL=3
TOP_K=5

DMODEL=64
NHEADS=8
ELAYERS=2
DLAYERS=1
DFF=2048

PATCH_LEN=16

# USE_DTW=--use_dtw
INVERSE=''

# ===== 想批量跑的组合 =====
DATASETS=(Weather) # 如果要测试不同数据集的话，在括号里添加即可
PRED_LENS=(96 192 336) # 如果要测试不同步长的话，在括号里添加即可

# ===== 各数据集路径与通道数（按你的目录改）=====
dataset_setup () {
    local ds="$1"
    case "$ds" in
      ETTh1) ROOT="./dataset/ETT-small/"; CSV="ETTh1.csv"; ENC_IN=7; DEC_IN=7; C_OUT=7;;
      ETTh2) ROOT="./dataset/ETT-small/"; CSV="ETTh2.csv"; ENC_IN=7; DEC_IN=7; C_OUT=7;;
      ETTm1) ROOT="./dataset/ETT-small/"; CSV="ETTm1.csv"; ENC_IN=7; DEC_IN=7; C_OUT=7;;
      ETTm2) ROOT="./dataset/ETT-small/"; CSV="ETTm2.csv"; ENC_IN=7; DEC_IN=7; C_OUT=7;;
      Weather)     ROOT="./dataset/weather/";       CSV="weather.csv";          ENC_IN=21;  DEC_IN=21;  C_OUT=21;;
      # Electricity) ROOT="./dataset/electricity/";   CSV="electricity.csv";      ENC_IN=321; DEC_IN=321; C_OUT=321;;
      # Exchange)    ROOT="./dataset/exchange_rate/"; CSV="exchange_rate.csv";    ENC_IN=8;   DEC_IN=8;   C_OUT=8;;
      # ILI)         ROOT="./dataset/ili/";           CSV="national_illness.csv"; ENC_IN=7;   DEC_IN=7;   C_OUT=7;;
      *) echo "未知数据集: $ds"; exit 1;;
    esac
}

# ============== 日志设置 ==============
TS="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="./logs/multi_${TS}"
mkdir -p "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/master.log"
log() { echo "[$(date +'%F %T')] $*" | tee -a "${MASTER_LOG}"; }

trap 'ecode=$?; [ $ecode -ne 0 ] && log "❌ 脚本异常退出，退出码=${ecode}"; exit $ecode' EXIT

run_with_log () {
    local baseline="$1"     # TimesNet / TimeMixer / DLinear
    local ds="$2"           # 数据集
    local pred="$3"         # 预测步长

    dataset_setup "${ds}"

    # 判断数据集是否为自定义类型
    local data_arg="${ds}"
    if [[ "${ds}" == "Weather" ]]; then
        data_arg="custom"
    fi

    local tag="${ds}_crc_${baseline,,}_m_pl${pred}_dm${DMODEL}"
    local log_file="${LOG_DIR}/${tag}.log"

    log "▶️  开始运行：${ds} / ${baseline} / pred_len=${pred}  (log: ${log_file})"

    # ------- 用 heredoc 回显完整命令（避免 printf 解析选项的问题） -------
    {
      echo "CMD @ $(date +'%F %T')"
      cat <<EOF
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model CRC \
  --baseline_model ${baseline} \
  --model_id ${tag} \
  --data ${data_arg} \
  --root_path ${ROOT} \
  --data_path ${CSV} \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${pred} \
  --enc_in ${ENC_IN} \
  --dec_in ${DEC_IN} \
  --c_out ${C_OUT} \
  --d_model ${DMODEL} \
  --n_heads ${NHEADS} \
  --e_layers ${ELAYERS} \
  --d_layers ${DLAYERS} \
  --d_ff ${DFF} \
  --patch_len "${PATCH_LEN}" \
  --batch_size ${BATCH} \
  --learning_rate ${LR} \
  --train_epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  ${USE_DTW:-} ${INVERSE} \
  --q_val ${Q_VAL} \
  --top_k ${TOP_K} \
  --des CRC_${baseline}_M_d${DMODEL}
EOF
    } | tee -a "${log_file}" >> "${MASTER_LOG}"

    # ------- 真正执行，同步写入 单模型日志 + 总日志 -------
    set +e
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model CRC \
      --baseline_model "${baseline}" \
      --model_id "${tag}" \
      --data "${data_arg}" \
      --root_path "${ROOT}" \
      --data_path "${CSV}" \
      --features M \
      --seq_len "${SEQ_LEN}" \
      --label_len "${LABEL_LEN}" \
      --pred_len "${pred}" \
      --enc_in "${ENC_IN}" \
      --dec_in "${DEC_IN}" \
      --c_out "${C_OUT}" \
      --d_model "${DMODEL}" \
      --n_heads "${NHEADS}" \
      --e_layers "${ELAYERS}" \
      --d_layers "${DLAYERS}" \
      --d_ff "${DFF}" \
      --patch_len "${PATCH_LEN}" \
      --batch_size "${BATCH}" \
      --learning_rate "${LR}" \
      --train_epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --use_gpu True \
      --gpu_type cuda \
      --gpu 0 \
      ${USE_DTW:-} ${INVERSE} \
      --q_val "${Q_VAL}" \
      --top_k "${TOP_K}" \
      --des "CRC_${baseline}_M_d${DMODEL}" \
      2> >(tee -a "${log_file}" >> "${MASTER_LOG}" >&2) \
      | tee -a "${log_file}" >> "${MASTER_LOG}"
    ecode=${PIPESTATUS[0]}
    set -e

    if [ $ecode -eq 0 ]; then
      log "✅ 完成：${ds} / ${baseline} / pred_len=${pred}"
    else
      log "❌ 失败：${ds} / ${baseline} / pred_len=${pred}，退出码=${ecode}（详见 ${log_file}）"
      return $ecode
    fi
}

# ===== 批量：数据集 × 预测步 × 基线 =====
for ds in "${DATASETS[@]}"; do
  for pl in "${PRED_LENS[@]}"; do
    run_with_log TimeXer "${ds}" "${pl}"
    run_with_log TimesNet "${ds}" "${pl}"
    run_with_log Autoformer "${ds}" "${pl}"
    run_with_log Informer  "${ds}" "${pl}"
    run_with_log PatchTST  "${ds}" "${pl}"
    run_with_log DLinear   "${ds}" "${pl}"
  done
done

log "🎉 全部批量运行完成。日志目录：${LOG_DIR}"
