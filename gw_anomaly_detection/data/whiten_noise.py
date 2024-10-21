#!/bin/python
import glob
import h5py as h5
import os
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ifo", default=None, type=str, help="The ifo, L1, H1 or K1.")
parser.add_argument("-s", "--idstart", default=None, type=int, help="The start id of the background noise.")
parser.add_argument("-e", "--idend", default=None, type=int, help="The end id of the background noise.")
parser.add_argument("-bg", "--bg-tlist-file", default=None, type=str, help="The hdf5 file of the time list of background noise.")
parser.add_argument("-af", "--asd-folder", default=None, type=str, help="The folder where the asd.txt files are stored.")
parser.add_argument("-of", "--output-folder", default=None, type=str, help="The folder to output the hdf5 files.")
args = parser.parse_args()

ifo = args.ifo
idstart = args.idstart
idend = args.idend
bg_tlist_file = args.bg_tlist_file
asd_folder = args.asd_folder
output_folder = args.output_folder

O3a_start, O3a_end = 1238166018, 1253977218
O3b_start, O3b_end = 1256655618, 1269363618
last_omicron_seg_H1_start, last_omicron_seg_H1_end = 1238834630, 1238834630+21869
last_omicron_seg_L1_start, last_omicron_seg_L1_end = 1238771446, 1238771446+1463
last_HL_end = min(last_omicron_seg_H1_end, last_omicron_seg_L1_end)
channels = {
    "H1": "H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
    "L1": "L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
}
window_length = 8
flow, fhigh = 30, 1500
resample = 4096
crop_length=1

def get_source(ifo, start, end):
    folder_prefix = f'/ceph/mirror/frames/O3/hoft_C01_clean_sub60Hz/{ifo}/{ifo[0]}-{ifo}_HOFT_CLEAN_SUB60HZ_C01'
    label0 = int(start/1e5)
    labelp = label0 - 1
    folderp = f"{folder_prefix}-{labelp}"
    if os.path.exists(folderp):
        labels = [labelp, label0]
    else:
        labels = [label0]
    if end >= (label0+1)*1e5:
        labels.append(label0+1)

    gwf_folders = [f'{folder_prefix}-{label}' for label in labels] 
    source = []
    for gwf_folder in gwf_folders:
        source.extend(glob.glob(f'{gwf_folder}/*.gwf'))

    source = sorted(source)
    return source

def select_asd(trigtime, asd_list):
    asd_times = [(int(file.split('/')[-1].split('-')[1]), int(file.split('/')[-1].split('-')[-1].split('.')[0])) for file in asd_list]
    for i in range(len(asd_times)):
        asd_sted = [asd_times[i][0], asd_times[i][0] + asd_times[i][1]]
        if trigtime >= asd_sted[0] and trigtime < asd_sted[1]:
            print(f"found asd for {trigtime}")
            asd = FrequencySeries.read(asd_list[i])

    return asd

def get_bg_ts(ifo, t0, window_length=8):
    bg_start, bg_end = t0, t0 + window_length
    source = get_source(ifo, bg_start, bg_end)
    ts = TimeSeries.read(
        source=source,
        channel=channels[ifo],
        start=bg_start,
        end=bg_end,
        gap='ignore',
    )
    return ts

def process(
        ts,
        asd,
        flow=30,
        fhigh=1500,
        resample=4096,
        crop_length=1,
):
    ts = ts.whiten(asd=asd)
    ts = ts.bandpass(flow, fhigh)
    ts = ts.crop(ts.t0.value + crop_length, ts.t0.value + ts.duration.value - crop_length)
    ts = ts.resample(resample)
    return ts

def main():
    # Read background noise's t0 and asd files
    # bg_tlist_file = f"/home/chiajui.chou/GW-anomaly-detection/makedata/{ifo}_bgseg_t_list_test.hdf5"
    with h5.File(bg_tlist_file, 'r') as f:
        keys = list(f.keys())
        bg_tlist = list(f[keys[0]][:])

    # asd_folder = f"/home/chiajui.chou/GW-anomaly-detection/data/asds/{ifo}/O3"
    asd_list = glob.glob(f"{asd_folder}/*.txt")
    asd_list = sorted(asd_list)

    # Whitening
    t_list = bg_tlist[idstart:idend]

    pts_list = []
    pt_list = []
    for t in t_list:
        try:
            ts = get_bg_ts(ifo, t)
            asd = select_asd(t, asd_list)
            pts = process(ts=ts, asd=asd)
            if ts.duration.value == window_length:
                pts_list.append(pts.to_value())
                pt_list.append(pts.t0.value)
        except Exception as e:
            print(f"{t}")
            print(str(e))


    # Write to hdf5
    if len(pts_list) != 0:
        output_h5 = f'{output_folder}/{ifo}_O3_bgts-{idstart}-{idend}.hdf5'
        with h5.File(output_h5, 'w') as w:
            data = np.stack(pts_list)
            w.create_dataset(
                'bgts',
                shape=data.shape,
                dtype='f8',
                data=data,
            )

            data = np.array(pt_list)
            w.create_dataset(
                'bgts_t0',
                shape=data.shape,
                dtype='f8',
                data=data,
            )

if __name__ == "__main__":
    main()
