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
print(len(glitch_seglist))

# Loading strain and asd into data cache.
data_cache = config['data']['data_cache']

# Waveform injection.

# Processing data.

# Writing processed data to s3 buckets.

# Gathering data.