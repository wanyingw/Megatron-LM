#!/bin/bash

# Same as bench_ns_gtp.sbatch, but installs the batched-SYRK emerging_optimizers
# build (PR #276) and runs with --use-syrk, for a side-by-side against the
# GEMM-path baseline numbers already produced by bench_ns_gtp.sbatch.
#
# GTP shards dense weights. At the production degree of 64 the group fits inside one
# GB200/GB300 NVLink domain, so NVLink is left enabled here: that is the transport a
# real run uses. On aws-cmh (4 GPUs/node) a 64-rank group needs 16 nodes.
#
# GTP degree and node count must agree: nodes * 4 (GPUs/node) == GTP. To try a smaller
# degree first, override both, e.g.
#   sbatch --nodes=2 --export=ALL,GTP=8 tools/muon_analysis/bench_ns_gtp_syrk.sbatch

#SBATCH -p batch
#SBATCH --nodes=16
#SBATCH --account=nemotron_sw_pre
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=bench_muon_ns_gtp_syrk

export NCCL_DEBUG=WARN

GTP=${GTP:-64}

ROOT_DIR="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/wanyingw/n4_gtp_muon"
REPO_DIR="${ROOT_DIR}/repos/deepak-muon-bench"
EO_DIR="${ROOT_DIR}/repos/.worktrees/eo-pr276-syrk"
IMAGE_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/nemotron4_prep/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh"

NAME="gtp_syrk"
LOGS_DIR="${ROOT_DIR}/muon_ns_bench/${NAME}/logs"
mkdir -p "${LOGS_DIR}"

# Cold Triton autotune+compile for the SYRK kernel is the dominant cost on a first run
# (minutes per distinct shape), not the NS iteration itself. Persist the cache on shared
# storage (same path the EGTP job warms) so every rank reuses compiled kernels instead of
# each of the 64 ranks recompiling redundantly.
export TRITON_CACHE_DIR="${ROOT_DIR}/muon_ns_bench/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

options=" \
    --group gtp \
    --modelled-world-size 12288 \
    --tp 1 --gtp ${GTP} --ep 64 --etp 1 --egtp 4 \
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
