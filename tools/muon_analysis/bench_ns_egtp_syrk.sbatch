#!/bin/bash

# Same as bench_ns_egtp.sbatch, but installs the batched-SYRK emerging_optimizers
# build (PR #276) and runs with --use-syrk, for a side-by-side against the
# GEMM-path baseline numbers already produced by bench_ns_egtp.sbatch.
#
# EGTP=4 fits on one aws-cmh node (4 GB300s), but in a real job the group can span the
# scale-out fabric, InfiniBand or RoCE depending on the cluster. NVLink, SHM and NVLS are
# disabled below so the collectives take the network path and the timings reflect that.

#SBATCH -p batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 0:40:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=bench_muon_ns_egtp_syrk

# Force the EGTP group off NVLink so it measures the IB path.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

ROOT_DIR="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/wanyingw/n4_gtp_muon"
REPO_DIR="${ROOT_DIR}/repos/deepak-muon-bench"
EO_DIR="${ROOT_DIR}/repos/.worktrees/eo-pr276-syrk"
IMAGE_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/nemotron4_prep/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh"

NAME="egtp_syrk"
LOGS_DIR="${ROOT_DIR}/muon_ns_bench/${NAME}/logs"
mkdir -p "${LOGS_DIR}"

# Cold Triton autotune+compile for the SYRK kernel is the dominant cost on a first run
# (minutes per distinct shape), not the NS iteration itself. Persist the cache on shared
# storage so every rank reuses compiled kernels instead of each recompiling redundantly,
# and so later runs (including the GTP job) start warm for shapes seen before.
export TRITON_CACHE_DIR="${ROOT_DIR}/muon_ns_bench/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

# The script sweeps every rank profile itself and reports the slowest, so nothing to pick.
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

# All 4 local tasks would otherwise `pip install -e` into the same site-packages at
# once and race each other (seen as a mid-install OSError). Share one container per
# node via --container-name and gate the install on local rank 0; the rest wait on a
# sentinel file inside that shared container.
INSTALL_FLAG="/tmp/eo_syrk_installed_${SLURM_JOB_ID}.flag"
run_cmd="
if [ \"\${SLURM_LOCALID}\" = \"0\" ]; then
    pip install -q -e ${EO_DIR} && touch ${INSTALL_FLAG}
else
    while [ ! -f ${INSTALL_FLAG} ]; do sleep 2; done
fi
python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py ${options}
"

srun -l \
    --container-image "${IMAGE_PATH}" \
    --container-name="ns_bench_syrk_${SLURM_JOB_ID}" \
    --container-mounts "/scratch:/scratch,/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
