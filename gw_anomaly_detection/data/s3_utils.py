#!/bin/python
import os
import boto3
import h5py
import toml
import numpy as np
import yaml
import re

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

# import gwpy
# from gwpy.segments import DataQualityFlag from gwpy.segments import Segment
# from gwpy.segments import SegmentList
# from gwpy.timeseries import TimeSeries
# from gwpy.timeseries import FrequencySeries

# import bilby
# from bilby.core.proir.analytical import Uniform, Cosine, Sine
# from bilby.gw.conversion import bilby_to_lalsimulation_spins

# import lal

# import pycbc
# from pycbc.waveform import get_sgburst_waveform
# from pycbc.waveform import get_td_waveform
# from pycbc.tpyes.timeseries import TimeSeries as pycbcts
# from pycbc.detector import Detector

class s3_session(boto3.Session):
    def set_client(
        self,
        access_key:str,
        secret_key:str,
        host_base:str,
    ):
        self.s3client = self.client(
            service_name='s3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=host_base,
        )

    def ls_bucket(self):
        response = self.s3client.list_buckets()
        for bucket in response['Buckets']:
            print(f"{bucket['Name']}")

    def ls_objects(
            self,
            Bucket:str,
            Prefix=None, 
            Silent=True
        ):
        if Prefix == None:
            response = self.s3client.list_objects_v2(
                Bucket=Bucket,
                Delimiter='/',
            )
        else:
            response = self.s3client.list_objects_v2(
                Bucket=Bucket,
                Delimiter='/',
                Prefix=Prefix,
            )      
        try:
            for prefix in response['CommonPrefixes']:
                print(f"DIR s3://{Bucket}/{prefix['Prefix']}")
        except KeyError:
            pass
        try:
            if not Silent:
                for content in response['Contents']:
                    print(f"s3://{Bucket}/{content['Key']}")
            file_list = [item['Key'] for item in response['Contents']]
        except KeyError:
            pass
        return file_list

def Getting_s3_config(
    yaml_file:str
    ):
    # Read the s3_config from the .yaml files
    
    with open(yaml_file, 'r') as f:
        cfg = f.read()
    
    configs = yaml.load(cfg, Loader=yaml.FullLoader)
    
    return configs

def Reading_file_dir(
    s3:s3_session,
    ifo:str, 
    CONFIGS:dict
    ):
    # Read the file list from the bucket and pass the file name list for picking out proper files
    
    ACCESS_KEY = CONFIGS['SESSION_CLIENT_CONFIGURATIONS']['ACCESS_KEY']
    SECRET_KEY = CONFIGS['SESSION_CLIENT_CONFIGURATIONS']['SECRET_KEY']
    HOST_BASE = CONFIGS['SESSION_CLIENT_CONFIGURATIONS']['HOST_BASE']
    
    BUCKET = CONFIGS['SESSION_BUCKET_CONFIGURATIONS']['BUCKET']
    PREFIX = CONFIGS['SESSION_BUCKET_CONFIGURATIONS']['COMMON_PREFIX']+ifo+'/'
    
    
    s3.set_client(ACCESS_KEY, SECRET_KEY, HOST_BASE)
    # s3.ls_bucket()
    file_list = s3.ls_objects(BUCKET, PREFIX)
    
    return file_list


def Is_overlapping(lower1, lower2, upper1, upper2):
    return not (upper1 < lower2 or upper2 < lower1)

def Computing_segments(
    file_list: str,
    start_time:int,
    end_time:int
    ):
    # Given the list of files, and the segment start time and end time we want, return the file list to download. 
    
    file_list_to_download = []
    for file_name in file_list:
        match = re.search(r'-(\d+)-(\d+)', file_name)
        if match:
            start = int(match.group(1))
            end = int(match.group(1)) + int(match.group(2))
            if Is_overlapping(start, start_time, end, end_time):
                file_list_to_download.append(file_name)
                
    return file_list_to_download

def Download_files(
    s3:s3_session,
    bucket:str,
    file_list: str, 
    data_dir:str
    ):
    # Download proper files, given dir and the file list. 
    
    for file_name in file_list:
        s3.s3client.download_file(bucket, file_name, data_dir+'/'+file_name.split('/')[-1])
    
    return 0

    


def Fetch_data(
    segment_start_time:int, 
    segment_end_time:int, 
    ifo:str,
    data_dir:str=None,
    s3_config_file:str = "s3_bucket_config.yaml"
    ):
    # Combined function for these file, reading the starting time
    
    if data_dir==None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'cache'))
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    s3 = s3_session()    
    CONFIGS = Getting_s3_config(s3_config_file)
    file_list = Reading_file_dir(s3, ifo, CONFIGS=CONFIGS)
    
    file_list = Computing_segments(file_list, segment_start_time, segment_end_time)
    
    print(file_list)
    
    Download_files(s3, CONFIGS['SESSION_BUCKET_CONFIGURATIONS']['BUCKET'], file_list, data_dir)
    
    
    


# def main():
    # ACCESS_KEY = "XRIYL1052YCQ125N75UM"
    # SECRET_KEY = "XES9mLNbxAWKZC1AUcYFlGd7ByyXXSYw5yB3UrwM"
    # HOST_BASE = "https://s3-west.nrp-nautilus.io"
    # s3 = s3_session()
    # s3.set_client(ACCESS_KEY, SECRET_KEY, HOST_BASE)
    # # s3.ls_bucket()
    # s3.ls_objects("cchou", "figs/")
    # s3.s3client.download_file("cchou", "figs/noise.png", "test/NOISE.png")

if __name__ == "__main__":
    
    # print(sys.path)
        
    # file_list = Reading_file_dir('L1', CONFIGS=Getting_s3_config("s3_bucket_config.yaml"))
    
    Fetch_data(1238543229, 1238543229, 'H1')
    
    # CONFIGS = Getting_s3_config()
    
    # ACCESS_KEY = CONFIGS['SESSION_CLIENT_CONFIGURATIONS']['ACCESS_KEY']
    # SECRET_KEY = CONFIGS['SESSION_CLIENT_CONFIGURATIONS']['SECRET_KEY']
    # HOST_BASE = CONFIGS['SESSION_CLIENT_CONFIGURATIONS']['HOST_BASE']
    
    # BUCKET = CONFIGS['SESSION_BUCKET_CONFIGURATIONS']['BUCKET']
    # PREFIX = CONFIGS['SESSION_BUCKET_CONFIGURATIONS']['PREFIX']
    
    # s3 = s3_session()
    # s3.set_client(ACCESS_KEY, SECRET_KEY, HOST_BASE)
    # s3.ls_bucket()
    # s3.ls_objects(BUCKET, PREFIX)
    
    
    
    # main()