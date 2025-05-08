#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=build
#SBATCH --error=_build.err
#SBATCH --output=_build.log
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=END
#SBATCH --mail-user=wlp9800@nyu.edu


IMAGE=oho-test
VERSION=1.0.2
DOCKER_URL="docker://thewillyp/${IMAGE}:master-${VERSION}"

# Build the Singularity image
singularity build --force /scratch/${USER}/images/${IMAGE}-${VERSION}-cpu.sif ${DOCKER_URL}-cpu
singularity build --force /scratch/${USER}/images/${IMAGE}-${VERSION}-gpu.sif ${DOCKER_URL}-gpu

# Create the overlay
singularity overlay create --size 20480 /scratch/${USER}/images/${IMAGE}-${VERSION}-cpu.sif
singularity overlay create --size 20480 /scratch/${USER}/images/${IMAGE}-${VERSION}-gpu.sif


# 10 GiB
