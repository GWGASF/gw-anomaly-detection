#!/bin/python
from gwpy.timeseries import TimeSeries

import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

def fetch_data(
        ifo: str,
        start: int,
        end: int,
        sample_rate: float = 16384,
        format: str = "gwf",
        host: str = "https://gwosc.org",
        data_cache: str = None,
):
    duration = end - start
    if duration <= 4096:
        t0_list = [start]
    else:
        t0_list = [i for i in range(start, end, 4096)]

    for t0 in t0_list:
        du = min(4096, end - t0)

        output_dir = f"{data_cache}/{ifo}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_name = f"{output_dir}/{ifo[0]}-{ifo}_GWOSC_STRAIN-{t0}-{du}.{format}"

        if os.path.exists(file_name):
            print(f"File already exists: {file_name}. Skipping download.")
            continue

        print(f"Fetching data from {t0} to {t0 + du}...")

        try:
            ts = TimeSeries.fetch_open_data(
                ifo=ifo,
                start=t0,
                end=t0 + du,
                sample_rate=sample_rate,
                format=format,
                host=host,
            )

            ts.write(
                file_name,
                format=format,
                overwrite=True
            )
            print(f"Data written to {file_name}.")
        except Exception as e:
            print(f"Failed to fetch {t0}-{t0 + du}: {e}")
            continue

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

