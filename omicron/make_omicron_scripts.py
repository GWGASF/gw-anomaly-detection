#!/bin/python3

import glob
import os

import numpy as np
from gwpy.segments import DataQualityFlag


class get_segs:
    def __init__(
        self,
        ifo: str,
        label: str,
        start: int,
        end: int,
    ) -> None:
        self.ifo = ifo
        self.label = f'{ifo}:{label}'
        self.start = start
        self.end = end

    def fetch_segs(self) -> None:
        segments = DataQualityFlag.query(
            self.label,
            self.start,
            self.end,
        )
        return segments

    def write_segs(self, file_name: str) -> None:
        self.segs.write(file_name)

    def read_segs(self, file_name: str) -> None:
        self.segs.read(file_name)

    def active_length(self, length_low=None):
        active = self.fetch_segs().active
        length_array = np.array(
            [[i, int(seg.end - seg.start)] for i, seg in enumerate(active)]
        )
        if length_low != None:
            filt = np.where(length_array[:, 1] >= length_low)
            length_array = length_array[filt]
        
        return length_array

    def active_start(self, length_low=None):
        active = self.fetch_segs().active
        length_array = np.array(
            [[i, int(seg.end - seg.start)] for i, seg in enumerate(active)]
        )
        start_array = np.array([int(seg.start) for seg in active])
        if length_low != None:
            filt = np.where(length_array[:, 1] >= length_low)
            start_array = start_array[filt]

        return start_array


def get_gwf(
    ifo: str,
    data_prefix: str,
    frame_type: str,
    gps_start: int,
    length: int,
):
    gps_folder_id = int(gps_start / 1e5)
    gwfs = []
    for gps_folder in range(gps_folder_id - 1, gps_folder_id + int(length / 1e5) + 2):
        gwf_list = glob.glob(
            f"{data_prefix}/{ifo}/{ifo[0]}-{ifo}_{frame_type}-{gps_folder}/*"
        )
        gwfs.extend(sorted(gwf_list))

    return gwfs


def main():
    # ifo = "L1"
    ifo = "H1"
    O3a_start, O3a_end = 1238112018, 1253923218
    O3b_start, O3b_end = 1256601618, 1269302418
    start, end = O3a_start, O3a_end

    # Generate the Omicorn scripts.
    data_prefix = f"/ceph/mirror/frames/O3/hoft_C01_clean_sub60Hz"
    omicron_prefix = "/home/chiajui.chou/GW-anomaly-detection/O3a"
    omicron_label = "GW"
    omicron_sample_frequency = 4096
    frametype = f"HOFT_CLEAN_SUB60HZ_C01"
    channels = f"{ifo}:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01"
    label = f"DCS-ANALYSIS_READY_C01:1"

    # Get the lists of the length and start gps time of the science segments. Here only the segments with the length >= 256 s are chosen.
    segs = get_segs(ifo, label, start, end)
    active_starts = segs.active_start(256)
    active_length = segs.active_length(256)[:, 1]

    for start, length in zip(active_starts, active_length):
        folder = f"{omicron_prefix}/{ifo}/{start}-{length}"
        gwfs = get_gwf(ifo, data_prefix, frametype, start, length)
        os.makedirs(folder)
        with open(f"{folder}/data_file.lcf", "w") as w:
            for line in gwfs:
                w.writelines(f"{line}\n")

        ini_text = f"""[{omicron_label}]
q-range = 3.3166 108.0
frequency-range = 32.0 2048.0
frametype = {frametype}
channels = {channels}
cluster-dt = 0.5
sample-frequency = {omicron_sample_frequency}
chunk-duration = 124
segment-duration = 64
overlap-duration = 4
mismatch-max = 0.2
snr-threshold = 5.0"""
        with open(f"{folder}/omicron_{omicron_label}.ini", "w") as w:
            w.write(ini_text)

        run_text = f"""omicron-process {omicron_label} \\
--gps {start} {start + length} \\
--ifo {ifo} \\
--config-file {folder}/omicron_{omicron_label}.ini \\
--output-dir {folder}/trigger_output \\
--cache-file {folder}/data_file.lcf \\
--verbose \\"""
        with open(f"{folder}/run_omicron.sh", "w") as w:
            w.write(run_text)


if __name__ == "__main__":
    main()
