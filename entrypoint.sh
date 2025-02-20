#!/bin/bash
set -e  # Exit on first error

# Initialize Micromamba properly
eval "$(/home/app/bin/micromamba shell hook -s bash)"
micromamba activate /home/app/micromamba/env
echo "Environment activated. Starting process..."

# Determine if running locally or in Kubernetes
if [[ -z "$K8S_ENV" ]]; then
    echo "Running locally, sourcing environment variables from gwenv.env"
    set -a
    source ./deployment/gwenv.env
    set +a
else
    echo "Running in Kubernetes, applying envsubst..."
fi

/home/app/micromamba/env/bin/envsubst < /home/app/opt/data_config.yaml > /home/app/opt/data_config.yaml.tmp

echo "Starting get_segments.py..."
python3 ./get_segments.py
echo "Finished get_segments.py."

# Adding debug for cli.py execution
echo "Starting cli.py..."
python3 -u ./cli.py
echo "Finished cli.py."
rm ./data_config.yaml.tmp

echo "Process finished successfully."
