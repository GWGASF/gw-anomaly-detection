#!/bin/python
import os
import yaml
import h5py
import numpy as np
from gw_anomaly_detection.data.segments import SegmentInfo
from gw_anomaly_detection.data.s3_utils import S3_session
from gw_anomaly_detection.data.waveforms import Waveforms
from gw_anomaly_detection.data.process import Process

# Loading data_config.yaml
with open("./data_config.yaml", "r") as file:
    config = yaml.safe_load(file)

ifos = config['data']['ifos']
data_cache = config['data']['data_cache']
asd_cache = config['data']['asd_cache']
injection = config['data']['injection']
output_dir = config['data']['output_dir']
keep_waveform = config['data']['gw_anomaly']['keep_waveform']

# Loading the information of the target segments.
background_segment_files = dict.fromkeys(ifos)
glitch_segment_files = dict.fromkeys(ifos)
bg_segs = dict.fromkeys(ifos)
glitch_segs = dict.fromkeys(ifos)
interval = dict.fromkeys(ifos)
for ifo in ifos:
    background_segment_files[ifo] = config['data']['background']['segment_files'][ifo]
    glitch_segment_files[ifo] = config['data']['glitch']['segment_files'][ifo]
    seg_info = SegmentInfo(ifo)
    bg_segs[ifo] = seg_info.read_segment_files(background_segment_files[ifo])
    glitch_segs[ifo] = seg_info.read_segment_files(glitch_segment_files[ifo])
    interval[ifo] = seg_info.whole_segment(background_segment_files[ifo])

    print(f"Number of background segments from {ifo}: {len(bg_segs[ifo])}, interval: {interval[ifo]}.")

# Loading strain and asd into data cache.
# s3 = S3_session(config['s3'])
# bucket = config['s3']['bucket']
# for ifo in ifos:
#     s3.fetch_data(
#             ifo=ifo,
#             start=interval[ifo][0],
#             end=interval[ifo][1],
#             bucket=bucket,
#             data_cache=data_cache,
#     )

# Generating Waveform.
waveform = config['data']['gw_anomaly']['waveform']
qm_file = config['data']['gw_anomaly']['qm_file']
sampling_frequency = config['data']['gw_anomaly']['sampling_frequency']
length = config['data']['gw_anomaly']['length']
number = 3
wav = Waveforms(
    ifos=ifos,
    length=length,
    sampling_frequency=sampling_frequency,
)
waveforms, params = wav.generate_waveforms(
    waveform,
    number,
    qm_file=qm_file,
)

# Rescaling and Injection.
flow = config['data']['processing']['flow']
fhigh = config['data']['processing']['fhigh']
resample = config['data']['processing']['resample']
crop_length = config['data']['processing']['crop_length']
target_snr_low = config['data']['gw_anomaly']['target_snr_low']
target_snr_high = config['data']['gw_anomaly']['target_snr_high']
proc = Process(
    ifos=ifos,
    data_cache=data_cache,
    asd_cache=asd_cache,
)
injected_ts, rescaled_waveforms, snrs = proc.inject(
    waveforms=waveforms,
    target_snr_low=target_snr_low,
    target_snr_high=target_snr_high,
    background_segments=bg_segs,
)

# Processing data.
timeseries = injected_ts
processed_data = proc.get_proccessed_data(
    timeseries=timeseries,
    background_segments=bg_segs,
    flow=flow,
    fhigh=fhigh,
    resample=resample,
    crop_length=crop_length,
)

# Write data to hdf5 files.
print(len(params))
print(params[0])
print(len(waveforms))
# print(waveforms[0])
print(len(snrs))
print(snrs[0])
print(len(processed_data))
# print(processed_data[0])
# for ifo in ifos:
#     network_snr = snrs[0]['network_snr']
#     snr = snrs[0][ifo]
#     processed_data[0][ifo].plot(title=f"{ifo}: {waveform}, SNR: {snr}, Network SNR: {network_snr}").savefig(f"./test/{ifo}-test.png")

# Uploading processed data to s3 buckets.

# Gathering data on s3 buckets.