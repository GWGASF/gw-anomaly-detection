import os
import re
import yaml
from pathlib import Path
from gw_anomaly_detection.data.s3_utils import S3_session

def load_data_config():
    config_path = os.path.join(os.path.dirname(__file__), "data_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def filter_files_by_range(file_list, start_gps, end_gps):
    filtered = []
    for f in file_list:
        match = re.search(r'-([0-9]+)-([0-9]+)\.txt$', f)
        if match:
            file_start = int(match.group(1))
            duration = int(match.group(2))
            if start_gps <= file_start <= end_gps:
                filtered.append(f)
    return filtered

def download_asd_files(s3: S3_session, ifo: str, start_gps: int, end_gps: int):
    local_dir = Path("asd_cache") / ifo
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nListing ASD files for {ifo}...")
    s3_files = s3.read_file_dir(ifo) or []

    if not s3_files:
        print(f"⚠️ No files returned for {ifo}. Skipping.")
        return

    filtered_files = filter_files_by_range(s3_files, start_gps, end_gps)
    print(f"Found {len(filtered_files)} files for {ifo} in range {start_gps} – {end_gps}")

    for file_key in filtered_files:
        filename = os.path.basename(file_key)
        local_path = local_dir / filename
        if local_path.exists():
            print(f"Skipping existing: {local_path}")
            continue
        s3_key = file_key.replace(f"s3://{s3.bucket}/", "")
        s3.download(s3_key, str(local_path))

def main():
    config = load_data_config()
    config['s3']['common_prefix'] = 'asds/O3a/'
    s3_config = config['s3']
    data_kind = config['data']['kind']
    ifos = config['data']['ifos']
    intervals = config['data'][data_kind]['sample_interval']

    s3 = S3_session(s3_config)

    for ifo in ifos:
        start = intervals[ifo]['start']
        end = intervals[ifo]['end']
        download_asd_files(s3, ifo, start, end)

    print("\nAll ASD downloads completed to ./asd_cache/")

if __name__ == "__main__":
    main()
