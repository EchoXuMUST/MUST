#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_help() {
  cat <<'USAGE'
Usage:
  bash run_deepresearch.sh dashscope "<question>"
  bash run_deepresearch.sh vllm-start-8b [model_path]
  bash run_deepresearch.sh vllm-run-8b "<question>" [model_name_or_path]
  bash run_deepresearch.sh eval /path/to/dataset.(json|jsonl|parquet) [--out /path/to/report.json] [--mock] [--image-root /path/to/images]

Modes:
  dashscope      Run DeepResearch via DashScope API.
  vllm-start-8b  Start local vLLM for Qwen3-VL-8B default on GPUs 8,9 with TP=2.
  vllm-run-8b    Run query against local vLLM, default model=Qwen/Qwen3-VL-8B-Instruct.
  eval           Run eval_vqa.py with passed arguments.

Required env for dashscope mode:
  DASHSCOPE_API_KEY

Examples:
  # 8B two-card default (GPU 8,9)
  bash run_deepresearch.sh vllm-start-8b
  bash run_deepresearch.sh vllm-start-8b /data/models/Qwen3-VL-8B-Instruct
  bash run_deepresearch.sh vllm-run-8b "请分析图文问答样例"

OOM tuning envs for vLLM mode (optional):
  VLLM_GPU_MEMORY_UTILIZATION (default 0.8)
  VLLM_MAX_MODEL_LEN (default 4096)
  VLLM_MAX_NUM_SEQS (default 1)
  VLLM_MAX_NUM_BATCHED_TOKENS (default 1024)
  VLLM_LIMIT_MM_PER_PROMPT (default {"image":1})

OOM tuning envs for API client (optional):
  API_MAX_TOKENS (default 128)
USAGE
}

ensure_python() {
  command -v python >/dev/null 2>&1 || {
    echo "[ERROR] python not found in PATH"
    exit 1
  }
}

ensure_vllm() {
  if ! python -c 'import vllm' >/dev/null 2>&1; then
    echo "[ERROR] vllm is not installed. Please install requirements first."
    exit 1
  fi
}

run_dashscope() {
  local question="${1:-}"
  [[ -n "$question" ]] || { echo "[ERROR] missing question"; exit 1; }

  : "${DASHSCOPE_API_KEY:?Please set DASHSCOPE_API_KEY}"
  export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
  export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
  export MODEL_NAME="${MODEL_NAME:-qwen-vl-max-latest}"

  ensure_python
  echo "[INFO] DashScope model=${MODEL_NAME}"
  python main.py "$question"
}

start_vllm_server() {
  local model_path="$1"
  local tp_size="$2"
  local gpu_ids="$3"
  local port="${VLLM_PORT:-8000}"
  local gpu_mem_util="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
  local max_model_len="${VLLM_MAX_MODEL_LEN:-4096}"
  local max_num_seqs="${VLLM_MAX_NUM_SEQS:-1}"
  local max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-1024}"
  local mm_limit="${VLLM_LIMIT_MM_PER_PROMPT:-}"
  if [[ -z "$mm_limit" ]]; then
    mm_limit='{"image":1}'
  fi

  ensure_python
  ensure_vllm

  export CUDA_VISIBLE_DEVICES="$gpu_ids"

  cat <<INFO
[INFO] Starting vLLM
[INFO] model_path=${model_path}
[INFO] tensor_parallel_size=${tp_size}
[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
[INFO] port=${port}
[INFO] gpu_memory_utilization=${gpu_mem_util}
[INFO] max_model_len=${max_model_len}
[INFO] max_num_seqs=${max_num_seqs}
[INFO] max_num_batched_tokens=${max_num_batched_tokens}
[INFO] limit_mm_per_prompt=${mm_limit}
INFO

  python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name "$model_path" \
    --host 0.0.0.0 \
    --port "$port" \
    --tensor-parallel-size "$tp_size" \
    --gpu-memory-utilization "$gpu_mem_util" \
    --max-model-len "$max_model_len" \
    --max-num-seqs "$max_num_seqs" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --limit-mm-per-prompt "$mm_limit"
}

run_vllm_start_8b() {
  local model_path="${1:-${MODEL_NAME_8B:-Qwen/Qwen3-VL-8B-Instruct}}"
  local tp_size="${VLLM_TP_SIZE_8B:-2}"
  local gpu_ids="${CUDA_VISIBLE_DEVICES_8B:-8,9}"

  start_vllm_server "$model_path" "$tp_size" "$gpu_ids"
}

run_vllm_query_8b() {
  local question="${1:-}"
  local model_name_arg="${2:-}"
  [[ -n "$question" ]] || { echo "[ERROR] missing question"; exit 1; }

  local port="${VLLM_PORT:-8000}"
  local model_name="${MODEL_NAME_8B:-Qwen/Qwen3-VL-8B-Instruct}"
  [[ -n "$model_name_arg" ]] && model_name="$model_name_arg"

  export OPENAI_BASE_URL="http://127.0.0.1:${port}/v1"
  export OPENAI_API_KEY="EMPTY"
  export MODEL_NAME="$model_name"

  ensure_python
  echo "[INFO] Query model=${MODEL_NAME} via ${OPENAI_BASE_URL}"
  python main.py "$question"
}

run_eval() {
  ensure_python
  python eval_vqa.py "$@"
}

main() {
  local mode="${1:-help}"
  shift || true

  case "$mode" in
    dashscope) run_dashscope "$@" ;;
    vllm-start|vllm-start-8b) run_vllm_start_8b "$@" ;;
    vllm-run-8b) run_vllm_query_8b "$@" ;;
    vllm-run) run_vllm_query_8b "$@" ;;
    eval) run_eval "$@" ;;
    -h|--help|help) print_help ;;
    *) echo "[ERROR] Unknown mode: $mode"; print_help; exit 1 ;;
  esac
}

main "$@"
