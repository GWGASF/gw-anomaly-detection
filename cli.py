#!/bin/python
import yaml
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
output_file = config['data']['output_file']
keep_waveform = config['data']['gw_anomaly']['keep_waveform']

# Loading the information of the target segments.
background_segment_files = dict.fromkeys(ifos)
glitch_segment_files = dict.fromkeys(ifos)
bg_segs = dict.fromkeys(ifos)
glitch_info_files = dict.fromkeys(ifos)
glitch_segs = dict.fromkeys(ifos)
interval = dict.fromkeys(ifos)
for ifo in ifos:
    background_segment_files[ifo] = config['data']['background']['segment_files'][ifo]
    glitch_info_files[ifo] = config['data']['glitch']['info_files'][ifo]
    glitch_segment_files[ifo] = config['data']['glitch']['segment_files'][ifo]
    seg_info = SegmentInfo(ifo)
    bg_segs[ifo] = seg_info.read_segment_files(background_segment_files[ifo])
    interval[ifo] = seg_info.whole_segment(background_segment_files[ifo])
    print(f"Number of background segments from {ifo}: {len(bg_segs[ifo])}, interval: {interval[ifo]}.")

    glitch_segs[ifo] = seg_info.read_segment_files(glitch_segment_files[ifo])
    interval[ifo] = seg_info.whole_segment(glitch_segment_files[ifo])
    print(f"Number of glitch segments from {ifo}: {len(glitch_segs[ifo])}, interval: {interval[ifo]}.")

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

# Processing data.
proc = Process(
    ifos=ifos,
    data_cache=data_cache,
    asd_cache=asd_cache,
)
if injection:
    # Generating Waveform.
    number = 10
    waveform = config['data']['gw_anomaly']['waveform']
    qm_file = config['data']['gw_anomaly']['qm_file']
    sampling_frequency = config['data']['gw_anomaly']['sampling_frequency']
    length = config['data']['gw_anomaly']['length']
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
    injected_ts, rescaled_waveforms, snrs = proc.inject(
        waveforms=waveforms,
        target_snr_low=target_snr_low,
        target_snr_high=target_snr_high,
        background_segments=bg_segs,
    )

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
    proc.write_injection_data(
        output_file=output_file,
        waveform_parameters=params,
        snrs=snrs,
        processed_data=processed_data,
        processed_waveforms=processed_waveforms,
    )
else:
    seg_start = 0
    seg_end = 1
    # noise_segments = bg_segs
    noise_segments = glitch_segs
    proc_noise = proc.get_processed_noise(
        noise_segments=noise_segments,
        seg_start=seg_start,
        seg_end=seg_end,
    )
    for ifo in ifos:
        print(ifo)
        print(len(proc_noise[ifo]))
        # print(proc_noise[ifo])

    # proc.write_noise_data(
    #     kind,
    #     output_file=output_file,
    #     processed_noise=proc_noise,
    #     glitch_info_files=glitch_info_files,
    # )

# Uploading processed data to s3 buckets.

# Gathering data on s3 buckets.