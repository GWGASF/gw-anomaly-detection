import sys
import subprocess
import yaml
from pathlib import Path

def create_job_yaml(ifo, sid, eid):
    ifo = ifo.upper()
    job_suffix = f"{ifo.lower()}-{sid}-{eid}"
    job_name = f"gwgasf-asds-{job_suffix}"

    job_dict = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": job_name,
                            "image": "dfredin/gwgasf:asdtest",
                            "command": [
                                "conda", "run", "--no-capture-output", "-n", "gwtesting", "python"
                            ],
                            "args": [
                                "run_asd.py", "--ifo", ifo, "--sid", str(sid), "--eid", str(eid)
                            ],
                            "resources": {
                                "limits": {
                                    "memory": "32Gi",
                                    "cpu": "6000m"
                                },
                                "requests": {
                                    "memory": "8Gi",
                                    "cpu": "1500m"
                                }
                            }
                        }
                    ],
                    "restartPolicy": "Never"
                }
            },
            "backoffLimit": 5
        }
    }

    return job_name, job_dict

def write_yaml_file(job_name, job_dict):
    filename = Path(f"{job_name}.yaml")
    with open(filename, "w") as f:
        yaml.dump(job_dict, f)
    return filename

def submit_job(yaml_file, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would apply: {yaml_file}")
    else:
        subprocess.run(["kubectl", "apply", "-f", str(yaml_file)], check=True)

def submit_single(ifo, sid, eid, dry_run=False, clean=False):
    job_name, job_yaml = create_job_yaml(ifo, sid, eid)
    yaml_file = write_yaml_file(job_name, job_yaml)
    submit_job(yaml_file, dry_run=dry_run)
    if clean and not dry_run:
        yaml_file.unlink()

def submit_batch(ifo, start_sid, end_eid, step, dry_run=False, clean=False):
    current = start_sid
    while current + step <= end_eid:
        submit_single(ifo, current, current + step, dry_run, clean)
        current += step
    if current < end_eid:
        submit_single(ifo, current, end_eid, dry_run, clean)

    print(f"{'[DRY RUN] ' if dry_run else ''}Batch complete from {start_sid} to {end_eid} in steps of {step}.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python submit_asd_job.py <ifo: H1|L1> <sid> <eid> [--batch] [--step N] [--dry-run] [--clean]")
        sys.exit(1)

    ifo = sys.argv[1]
    sid = int(sys.argv[2])
    eid = int(sys.argv[3])
    batch_mode = "--batch" in sys.argv
    dry_run = "--dry-run" in sys.argv
    clean = "--clean" in sys.argv

    # Handle optional --step
    if "--step" in sys.argv:
        try:
            step_index = sys.argv.index("--step")
            step = int(sys.argv[step_index + 1])
        except (IndexError, ValueError):
            print("Error: --step flag must be followed by an integer")
            sys.exit(1)
    else:
        step = 100

    if batch_mode:
        submit_batch(ifo, sid, eid, step, dry_run=dry_run, clean=clean)
    else:
        submit_single(ifo, sid, eid, dry_run=dry_run, clean=clean)
