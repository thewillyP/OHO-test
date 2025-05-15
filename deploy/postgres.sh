#!/bin/bash
#SBATCH --job-name=oho_db
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output="/scratch/wlp9800/logs/%x-%j.out"
#SBATCH --error="/scratch/wlp9800/logs/%x-%j.err"


source ~/.secrets/env.sh

# Ensure the postgres directory exists
mkdir -p "$DB_PATH_OHO"

# Set or replace DB_HOST in env.sh with the current hostname
sed -i '/^export DB_HOST_OHO=/d' ~/.secrets/env.sh
echo "export DB_HOST_OHO=$(hostname)" >> ~/.secrets/env.sh

singularity run --containall --cleanenv \
  --env POSTGRES_USER="$POSTGRES_USER" \
  --env POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  --env POSTGRES_DB="$POSTGRES_DB_OHO" \
  --env PGPORT="$PGPORT" \
  --bind "$DB_PATH_OHO":/var \
  docker://postgres:17


