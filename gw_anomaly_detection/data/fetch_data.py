#!/bin/python
from gwpy.timeseries import TimeSeries
import os
import requests

def fetch_strain_list(run: str, detector: str, gps_start: int, gps_end: int):
    """
    Query GWOSC for available strain files for a detector in the given GPS range.
    Returns a list of strain file info dicts.
    """
    url = f"https://gwosc.org/archive/links/{run}/{detector}/{gps_start}/{gps_end}/json/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("strain", [])
    except Exception as e:
        print(f"[ERROR] Failed to fetch file list from {url}: {e}")
        return []

def download_file(url, out_path):
    """
    Downloads a file and verifies it is not truncated.
    Returns True if successful, False if failed or incomplete.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            content_length = int(r.headers.get('Content-Length', 0))

            with open(out_path, "wb") as f_out:
                for chunk in r.iter_content(chunk_size=8192):
                    f_out.write(chunk)

        actual_size = os.path.getsize(out_path)
        if content_length and actual_size < content_length:
            print(f"[WARN] Truncated file: {out_path} ({actual_size} < {content_length}). Deleting.")
            os.remove(out_path)
            return False

        return True

    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return False

def fetch_data(
        ifo: str,
        start: int,
        end: int,
        sample_rate: float = 16384,
        format: str = "hdf5",
        run: str = "O3a_16KHZ_R1",
        data_cache: str = "./data_cache"
):
    """
    Downloads all available 16 kHz HDF5 strain files for a given detector and GPS interval.
    Includes integrity checks and transparent logs.
    """
    os.makedirs(os.path.join(data_cache, ifo), exist_ok=True)
    strain_files = fetch_strain_list(run, ifo, start, end)

    # Filter only .hdf5 files
    hdf5_files = [
        f for f in strain_files
        if f.get("url", "").endswith(".hdf5")
    ]

    print(f"[INFO] Found {len(hdf5_files)} total HDF5 files for {ifo} in GPS range {start}–{end}.")

    if not hdf5_files:
        print(f"[WARNING] No HDF5 files to process.")
        return 1

    downloaded = 0
    skipped = 0
    failed = 0

    for f in hdf5_files:
        url = f["url"]
        filename = url.split("/")[-1]
        out_path = os.path.join(data_cache, ifo, filename)

        if os.path.exists(out_path):
            # Check size for safety
            expected_size = int(requests.head(url).headers.get('Content-Length', 0))
            actual_size = os.path.getsize(out_path)
            if expected_size and actual_size < expected_size:
                print(f"[WARN] Detected incomplete file: {filename}. Re-downloading.")
                os.remove(out_path)
            else:
                print(f"[SKIP] {filename} already exists.")
                skipped += 1
                continue

        print(f"[DOWNLOAD] {filename} from {url}")
        success = download_file(url, out_path)
        if success:
            print(f"[SAVED] {filename} to {out_path}")
            downloaded += 1
        else:
            print(f"[RETRY NEEDED] {filename} failed or was truncated.")
            failed += 1

    print(f"[DONE] {downloaded} downloaded, {skipped} skipped, {failed} failed for {ifo}.")
    return 0






def override_config_with_env(config):
    # Override total_interval
    for ifo in ['H1', 'L1']:
        start_env = os.getenv(f"{ifo}_TOTAL_START")
        end_env = os.getenv(f"{ifo}_TOTAL_END")
        if start_env and end_env:
            config['data']['total_interval'][ifo]['start'] = int(start_env)
            config['data']['total_interval'][ifo]['end'] = int(end_env)

    # Override sample_interval for all kinds that define it
    kinds = ['background', 'glitch', 'injection']
    for kind in kinds:
        if kind in config['data']:
            for ifo in ['H1', 'L1']:
                start_env = os.getenv(f"{ifo}_KIND_START")
                end_env = os.getenv(f"{ifo}_KIND_END")
                if start_env and end_env:
                    start = int(start_env)
                    end = int(end_env)
                    config['data'][kind]['sample_interval'][ifo]['start'] = start
                    config['data'][kind]['sample_interval'][ifo]['end'] = end

            # After updating intervals, update segment_files for this kind
            segment_files = {}
            for ifo in config['data'][kind]['ifos']:
                start = config['data'][kind]['sample_interval'][ifo]['start']
                end = config['data'][kind]['sample_interval'][ifo]['end']
                duration = end - start
                segment_files[ifo] = [f"./segments/O3a/{ifo}-{kind}_samples-{start}-{duration}.segwizard"]
            config['data'][kind]['segment_files'] = segment_files

    return config

