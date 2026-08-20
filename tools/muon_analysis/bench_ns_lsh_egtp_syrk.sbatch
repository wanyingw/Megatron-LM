#!/bin/bash

# bench_ns_lsh.py with --use-syrk on the EGTP axis: layer_sharded / layer_sharded_batched
# (and the reused duplicated/distributed baselines) all routed through the batched-SYRK
# Triton kernel (PR #276). Compare against bench_ns_lsh_egtp.sbatch (GEMM baseline).
#
# Uses PYTHONPATH straight into the PR276 worktree rather than `pip install -e`: EO has
# no compiled extension, and installing from 4 ranks into the same site-packages at once
# races (see bench_ns_egtp_syrk.sbatch history). PYTHONPATH sidesteps that entirely.

#SBATCH -p batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 0:30:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=bench_muon_ns_lsh_egtp_syrk

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

ROOT_DIR="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/wanyingw/n4_gtp_muon"
REPO_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/wanyingw/n4_gtp_muon/repos/deepak-muon-bench"
EO_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/wanyingw/n4_gtp_muon/repos/.worktrees/eo-pr276-syrk"
IMAGE_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/nemotron4_prep/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh"

NAME="lsh_egtp_syrk"
LOGS_DIR="${ROOT_DIR}/muon_ns_bench/${NAME}/logs"
mkdir -p "${LOGS_DIR}"

# Same cache the plain (non-lsh) syrk run warmed for these two EGTP shapes, so this
# run should mostly hit a warm Triton cache rather than cold-compiling again.
export TRITON_CACHE_DIR="${ROOT_DIR}/muon_ns_bench/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

options=" \
    --group egtp \
    --modelled-world-size 12288 \
    --tp 1 --gtp 64 --ep 64 --etp 1 --egtp 4 \
    --shard-latent-proj \
    --num-ns-steps 16 \
    --coefficient-type polar_express \
    --iters 20 \
    --warmup 5 \
    --use-syrk "

run_cmd="cd ${REPO_DIR} && PYTHONPATH=${EO_DIR}:\$(pwd) python -u tools/muon_analysis/bench_ns_lsh.py ${options}"

srun -l \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/scratch:/scratch,/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
