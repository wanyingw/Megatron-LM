#!/bin/bash

# bench_ns_lsh.py (layer_sharded / layer_sharded_batched vs duplicated/distributed)
# on the EGTP axis. Baseline (GEMM) run; see bench_ns_lsh_egtp_syrk.sbatch for the
# batched-SYRK counterpart at the same shapes.
#
# EGTP=4 fits on one aws-cmh node (4 GB300s), but in a real job the group can span the
# scale-out fabric, InfiniBand or RoCE depending on the cluster. NVLink, SHM and NVLS are
# disabled below so the collectives take the network path and the timings reflect that.

#SBATCH -p batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 0:15:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=bench_muon_ns_lsh_egtp

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

ROOT_DIR="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/wanyingw/n4_gtp_muon"
REPO_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/wanyingw/n4_gtp_muon/repos/deepak-muon-bench"
EO_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/wanyingw/n4_gtp_muon/repos/.worktrees/eo-r0.4.0"
IMAGE_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/nemotron4_prep/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh"

NAME="lsh_egtp"
LOGS_DIR="${ROOT_DIR}/muon_ns_bench/${NAME}/logs"
mkdir -p "${LOGS_DIR}"

DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

options=" \
    --group egtp \
    --modelled-world-size 12288 \
    --tp 1 --gtp 64 --ep 64 --etp 1 --egtp 4 \
    --shard-latent-proj \
    --num-ns-steps 16 \
    --coefficient-type polar_express \
    --iters 20 \
    --warmup 5 "

run_cmd="cd ${REPO_DIR} && PYTHONPATH=${EO_DIR}:\$(pwd) python -u tools/muon_analysis/bench_ns_lsh.py ${options}"

srun -l \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/scratch:/scratch,/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
