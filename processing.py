#!/bin/python
import os
import yaml
from gw_anomaly_detection.data.segments import SegmentInfo
from gw_anomaly_detection.data.s3_utils import S3_session
from gw_anomaly_detection.data.waveforms import Waveforms
from gw_anomaly_detection.data.process import Process

# Loading data_config.yaml
with open("./data_config.yaml", "r") as file:
    config = yaml.safe_load(file)

ifo = config['data']['ifo']
data_cache = config['data']['data_cache']
asd_cache = config['data']['asd_cache']
injection = config['data']['injection']

# Loading the information of the target segments.
background_segment_files = config['data']['background']['segment_files']
glitch_segment_files = config['data']['glitch']['segment_files']

seg_info = SegmentInfo(ifo)
bg_segs = seg_info.read_segment_files(background_segment_files)
interval = seg_info.whole_segment(background_segment_files)

print(f"Number of background segments from {ifo}: {len(bg_segs)}")

# Loading strain and asd into data cache.
# s3 = S3_session(config['s3'])
# bucket = config['s3']['bucket']
# s3.fetch_data(
#         ifo=ifo,
#         start=interval[0],
#         end=interval[1],
#         bucket=bucket,
#         data_cache=data_cache,
# )

# Generating Waveform.
waveform = config['data']['gw_anomaly']['waveform']
qm_file = config['data']['gw_anomaly']['qm_file']
sampling_frequency = config['data']['gw_anomaly']['sampling_frequency']
length = config['data']['gw_anomaly']['length']
number = 1
wav = Waveforms(
    ifos=[ifo],
    length=length,
    sampling_frequency=sampling_frequency,
)
waveforms = wav.generate_waveforms(
    waveform,
    number,
    qm_file=qm_file,
)

# Injection.
segment = bg_segs[0]
proc = Process(ifo)
ts = proc.get_ts(
    data_cache=data_cache,
    segment=segment,
)
sig = waveforms[0][ifo]
sig.t0 = ts.t0
inj_ts = ts.inject(sig)

# Processing data.

# Uploading processed data to s3 buckets.

# Gathering data.