#!/bin/python
import os
import yaml
from gwpy.segments import SegmentList
from gw_anomaly_detection.data.segments import segment_info
from gw_anomaly_detection.data.s3_utils import s3_session

# Loading data_config.yaml
with open("./data_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Loading the information of the target segments.
background_segment_file = config['data']['background']['segment_file']
glitch_segment_file = config['data']['glitch']['segment_file']

background_seglist = SegmentList.read(background_segment_file, format='segwizard')
glitch_seglist = SegmentList.read(glitch_segment_file, format='segwizard')

print(len(background_seglist))
background_starts = [seg.start.gpsSeconds for seg in background_seglist]
background_ends = [seg.end.gpsSeconds for seg in background_seglist]
background_interval = (min(background_starts), max(background_ends))
print(background_interval)

# Loading strain and asd into data cache.
data_cache = config['data']['data_cache']
s3 = s3_session(config['s3'])
s3.fetch_data(
        ifo=config['data']['ifo'],
        start=background_interval[0],
        end=background_interval[1],
        bucket=config['s3']['bucket'],
        data_cache=config['data']['data_cache'],
)

# Waveform injection.

# Processing data.

# Writing processed data to s3 buckets.

# Gathering data.