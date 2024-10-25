#!/bin/python
import h5py
from gwpy.segments import SegmentList

def get_background_segments(
        ifo: str,
        segment_file: str,
        glitch_info_file: str,
        glitch_window_length: float=2,
):
    with h5py.File(glitch_info_file, 'r') as f:
        glitch_times = f['glitch_info']['time'][:]

    return glitch_times