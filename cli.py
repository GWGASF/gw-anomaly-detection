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
if keep_waveform:
    timeseries = rescaled_waveforms
    processed_waveforms = proc.get_proccessed_data(
        timeseries=timeseries,
        background_segments=bg_segs,
        flow=flow,
        fhigh=fhigh,
        resample=resample,
        crop_length=crop_length,
    )

# Write data to hdf5 files.
param_names = list(params[0].keys())
param_formats = []
for name in param_names:
    if isinstance(params[0][name], float) or isinstance(params[0][name], int):
        param_formats.append('f8')
    if isinstance(params[0][name], str):
        param_formats.append(h5py.string_dtype(encoding="ascii"))

snr_names = list(snrs[0].keys())
snr_formats = ['f8' for i in range(len(snr_names))]
names = param_names + snr_names
formats = param_formats + snr_formats
dt = np.dtype({'names': names, 'formats': formats})
param_data = np.array(
    [tuple(param.values()) + tuple(snr.values())
        for param, snr in zip(params, snrs)],
    dtype=dt,
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with h5py.File(f"{output_dir}/test_data.hdf5", 'w') as w:
    w.create_dataset(
        'waveform_parameters',
        shape=param_data.shape,
        dtype=dt,
        data=param_data,
    )

    ifos = list(processed_data[0].keys())
    proc_data = dict.fromkeys(ifos)
    t0_data = dict.fromkeys(ifos)
    for ifo in ifos:
        proc_data[ifo] = np.array([data[ifo] for data in processed_data])
        t0_data[ifo] = np.array([data[ifo].t0.value for data in processed_data])
        sample_rate = processed_data[0][ifo].sample_rate.value
        proc_dset = w.create_dataset(
            ifo,
            shape=proc_data[ifo].shape,
            dtype=proc_data[ifo].dtype,
            data=proc_data[ifo]
        )
        proc_dset.attrs['sample_rate'] = sample_rate
        w.create_dataset(
            f"t0_{ifo}",
            shape=t0_data[ifo].shape,
            dtype=t0_data[ifo].dtype,
            data=t0_data[ifo]
        )

    if keep_waveform:
        ifos = list(processed_waveforms[0].keys())
        waveform_data = dict.fromkeys(ifos)
        for ifo in ifos:
            waveform_data[ifo] = np.array([waveform[ifo] for waveform in processed_waveforms])
            channel = str(processed_waveforms[0][ifo].channel)
            print(waveform_data[ifo][0])
            waveform_dset = w.create_dataset(
                f"waveform_{ifo}",
                shape=waveform_data[ifo].shape,
                dtype=waveform_data[ifo].dtype,
                data=waveform_data[ifo]
            )
            waveform_dset.attrs['channel'] = channel

# Uploading processed data to s3 buckets.

# Gathering data on s3 buckets.