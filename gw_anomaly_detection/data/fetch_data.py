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
        data_cache: str=None,
):
    t0_list = [i for i in range(start, end, 4096)]
    print(t0_list)
    return 0