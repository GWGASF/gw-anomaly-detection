# GW Anomaly Detection

## Overview

This repository, **gw-anomaly-detection**, is part of the **GWGASF** collaboration. It focuses on generating simulated data from the [LIGO O3a archive](https://gwosc.org/O3/O3a/), aiming to support gravitational wave anomaly detection.

The repository can be run locally as a Docker container or on a Kubernetes cluster. Environmental variables are managed through the `gwenv.env` file for local runs and corresponding environment variables in the Kubernetes deployment YAML files.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Kubernetes](https://kubernetes.io/docs/setup/)
- SSH key setup (`~/.ssh/id_ed25519`) for GitLab access
- Access to GitLab Container Registry at `gitlab-registry.nrp-nautilus.io`

---

## Cloning the Repository

First, clone the repository to your local machine:

```bash
git pull <repository-url>
```

---

## Building the Docker Image

Navigate to the directory containing the local repository and run the following commands:

```bash
cd <path-to-local-repo>

export DOCKER_BUILDKIT=1

# Start SSH agent and add SSH key
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_ed25519

# Build the Docker image
docker build --ssh default --no-cache -t \
  gitlab-registry.nrp-nautilus.io/gwgasf/gwgasf/gw-anomaly-detection:testinggw .

# Push the Docker image to GitLab Registry
docker push gitlab-registry.nrp-nautilus.io/gwgasf/gwgasf/gw-anomaly-detection:testinggw
```

---

## Running the Docker Container Locally

Ensure that the `gwenv.env` file is updated with the necessary environmental variables. Then run:

```bash
docker run --rm -it --env-file gwenv.env \
  gitlab-registry.nrp-nautilus.io/gwgasf/gwgasf/gw-anomaly-detection:testinggw \
  sh -c "/home/app/micromamba/env/bin/envsubst < /home/app/opt/data_config.yaml > /home/app/opt/data_config.yaml.tmp"
```

---

## Running on a Kubernetes Cluster

1. Navigate to the `deployment` directory:

```bash
cd deployment
```

2. Apply the deployment YAML file (ensure environmental variables are updated as needed):

```bash
kubectl create -f <name-of-yaml-file>
```

---

## Environmental Variables

- **Local Run:** Update the `gwenv.env` file with necessary environment variables.
- **Kubernetes Run:** Ensure the deployment YAML files contain the correct environmental variables.

---

## Notes

- Ensure SSH keys are correctly configured for GitLab access.
- Adjust environmental variables as per your deployment needs.
- The Docker image tag `testinggw` can be modified for different versions or testing purposes.

---

## Troubleshooting

- **SSH Issues:** Verify your SSH key is located at `~/.ssh/id_ed25519` and has appropriate permissions.
- **Docker Build Failures:** Ensure Docker is running and accessible. Re-run the build commands with `--no-cache` if needed.
- **Kubernetes Deployment Errors:** Validate the YAML file paths and syntax.

---

## License

[MIT License](LICENSE)

---

## Contact

For further questions, please contact:

- **Daniel Fredin**  
  Contributor to GWGASF Collaboration

---

*This README provides the necessary instructions for building, running, and deploying the GW Anomaly Detection repository locally and on a Kubernetes cluster.*

