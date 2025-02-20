import os
import yaml
import subprocess
import uuid

# Base Kubernetes Job template
JOB_TEMPLATE = """
apiVersion: batch/v1
kind: Job
metadata:
  name: gw-anomaly-job-{job_id}
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: gwanomaly
        image: gitlab-registry.nrp-nautilus.io/gwgasf/gwgasf/gw-anomaly-detection:data
        workingDir: /opt
        command: ["/bin/sh", "-c"]
        args:
          - |
            cat <<EOF > /home/app/opt/data_config.yaml
{config_data}
            EOF
            python /home/app/opt/cli.py
"""

# Parallel job configurations
jobs_to_run = [
    {"kind": "background", "start_id": 0, "end_id": 20},
    {"kind": "background", "start_id": 20, "end_id": 40},
    # {"kind": "glitch", "start_id": 0, "end_id": 20},
    # {"kind": "glitch", "start_id": 20, "end_id": 40},
    # {"kind": "injection", "start_id": 0, "end_id": 20},
    # {"kind": "injection", "start_id": 20, "end_id": 40},
]

# Load base config file
with open("data_config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# Create and apply jobs dynamically
for job in jobs_to_run:
    job_id = str(uuid.uuid4())[:8]  # Unique job ID

    # Modify base config
    base_config["data"]["kind"] = job["kind"]
    base_config["data"][job["kind"]]["start_id"] = job["start_id"]
    base_config["data"][job["kind"]]["end_id"] = job["end_id"]

    # Convert modified config to YAML format and indent for shell script
    modified_config_yaml = yaml.dump(base_config, default_flow_style=False, indent=2)
    indented_config = "\n".join(["            " + line for line in modified_config_yaml.split("\n")])

    # Create Job YAML file with embedded config
    job_yaml = JOB_TEMPLATE.format(job_id=job_id, config_data=indented_config)
    job_file = f"job-{job_id}.yaml"
    with open(job_file, "w") as f:
        f.write(job_yaml)

    # Apply Job using kubectl
    subprocess.run(["kubectl", "apply", "-f", job_file])

    print(f"Deployed job {job_id} for kind={job['kind']} with start_id={job['start_id']} and end_id={job['end_id']}")

print("All jobs have been deployed!")
