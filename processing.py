#!/bin/python
import os
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
waveforms = wav.generate_waveforms(
    waveform,
    number,
    qm_file=qm_file,
)

# Rescaling and Injection.
flow = config['data']['processing']['flow']
fhigh = config['data']['processing']['fhigh']
target_snr_low = config['data']['gw_anomaly']['target_snr_low']
target_snr_high = config['data']['gw_anomaly']['target_snr_high']
proc = Process(
    ifos=ifos,
    data_cache=data_cache,
    asd_cache=asd_cache,
)
rescaled_waveforms, rescaled_snrs = proc.rescale(
    waveforms=waveforms,
    target_snr_low=target_snr_low,
    target_snr_high=target_snr_high,
    background_segments=bg_segs,
)

        # ts = proc.get_ts(ifo, segment)
        # signal.t0 = ts.t0
        # inj_ts = ts.inject(signal)
        # injs[ifo] = inj_ts
    # inj_tss.append(injs)


# Processing data.

# Uploading processed data to s3 buckets.

# Gathering data.