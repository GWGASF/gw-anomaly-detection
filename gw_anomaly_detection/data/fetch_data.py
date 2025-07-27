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

    return 0
